""" Training augmented model """
import torch.nn as nn
import torchvision
from tensorboardX import SummaryWriter
from config import RandomSearchConfig
import utils
from models.augment_cnn import AugmentCNN
from utils import *


config = RandomSearchConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)
logger.info("Logger is set - searching start")
logger.info("Torch version is: {}".format(torch.__version__))
logger.info("Torch_vision version is: {}".format(torchvision.__version__))


def main(genotype):

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # get data with meta info
    input_size, input_channels, n_classes, train_data, valid_data = utils.get_data(
        config.dataset, config.data_path, config.cutout_length, validation=True)

    criterion = nn.CrossEntropyLoss().to(device)
    use_aux = config.aux_weight > 0.
    model = AugmentCNN(input_size, input_channels, config.init_channels, n_classes, config.layers,
                       use_aux, genotype)
    # model size
    mb_params = utils.param_size(model)
    logger.info("Model size = {:.3f} MB".format(mb_params))
    model = nn.DataParallel(model, device_ids=config.gpus).to(device)

    # weights optimizer
    optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=config.momentum,
                                weight_decay=config.weight_decay)

    # get data loader
    if config.data_loader_type == 'Torch':
        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=config.batch_size,
                                                   shuffle=True,
                                                   num_workers=config.workers,
                                                   pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(valid_data,
                                                   batch_size=config.batch_size,
                                                   shuffle=False,
                                                   num_workers=config.workers,
                                                   pin_memory=True)
    elif config.data_loader_type == 'DALI':
        config.dataset = config.dataset.lower()
        if config.dataset == 'cifar10':
            from DataLoaders_DALI import cifar10
            train_loader = cifar10.get_cifar_iter_dali(type='train',
                                                       image_dir=config.data_path,
                                                       batch_size=config.batch_size,
                                                       num_threads=config.workers)
            valid_loader = cifar10.get_cifar_iter_dali(type='val',
                                                       image_dir=config.data_path,
                                                       batch_size=config.batch_size,
                                                       num_threads=config.workers)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)

    best_top1 = 0.
    if config.data_loader_type == 'DALI':
        len_train_loader = get_train_loader_len(config.dataset.lower(), config.batch_size, is_train=True)
    else:
        len_train_loader = len(train_loader)
    # training loop
    for epoch in range(config.epochs):
        lr_scheduler.step()
        drop_prob = config.drop_path_prob * epoch / config.epochs
        model.module.drop_path_prob(drop_prob)

        # training
        train(train_loader, model, optimizer, criterion, epoch)

        # validation
        cur_step = (epoch+1) * len_train_loader
        top1 = validate(valid_loader, model, criterion, epoch, cur_step)

        # save
        if best_top1 < top1:
            best_top1 = top1
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best)

        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    return best_top1


def train(train_loader, model, optimizer, criterion, epoch):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    def train_iter(X, y):
        N = X.size(0)

        optimizer.zero_grad()
        logits, aux_logits = model(X)
        loss = criterion(logits, y)
        if config.aux_weight > 0.:
            loss += config.aux_weight * criterion(aux_logits, y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len_train_loader - 1:
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch + 1, config.epochs, step, len_train_loader - 1, losses=losses,
                    top1=top1, top5=top5))

    if config.data_loader_type == 'DALI':
        len_train_loader = get_train_loader_len(config.dataset.lower(), config.batch_size, is_train=True)
    else:
        len_train_loader = len(train_loader)
    cur_step = epoch*len_train_loader
    cur_lr = optimizer.param_groups[0]['lr']
    logger.info("Epoch {} LR {}".format(epoch, cur_lr))
    writer.add_scalar('train/lr', cur_lr, cur_step)

    model.train()
    if config.data_loader_type == 'DALI':
        for step, data in enumerate(train_loader):
            X = data[0]["data"].cuda(async=True)
            y = data[0]["label"].squeeze().long().cuda(async=True)
            if config.cutout_length > 0:
                X = cutout_batch(X, config.cutout_length)
            train_iter(X, y)
            cur_step += 1
        train_loader.reset()
    else:
        for step, (X, y) in enumerate(train_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            train_iter(X, y)
            cur_step += 1
    logger.info("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))


def validate(valid_loader, model, criterion, epoch, cur_step):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    if config.data_loader_type == 'DALI':
        len_val_loader = get_train_loader_len(config.dataset.lower(), config.batch_size, is_train=False)
    else:
        len_val_loader = len(valid_loader)

    def val_iter(X, y):
        N = X.size(0)

        logits, _ = model(X)
        loss = criterion(logits, y)

        prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len_val_loader - 1:
            logger.info(
                "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch + 1, config.epochs, step, len_val_loader - 1, losses=losses,
                    top1=top1, top5=top5))

    model.eval()

    with torch.no_grad():
        if config.data_loader_type == 'DALI':
            for step, data in enumerate(valid_loader):
                X = data[0]["data"].cuda(async=True)
                y = data[0]["label"].squeeze().long().cuda(async=True)
                val_iter(X, y)
            valid_loader.reset()
        else:
            for step, (X, y) in enumerate(valid_loader):
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                val_iter(X, y)
    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/top1', top1.avg, cur_step)
    writer.add_scalar('val/top5', top5.avg, cur_step)
    logger.info("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))
    return top1.avg


if __name__ == "__main__":
    import genotypes as gt
    file = open(config.file)
    lines = file.readlines()
    best = 0
    best_ge = None
    for line in lines:
        logger.info('Now the genotype is:{0}'.format(line))
        ge = gt.from_str(line)
        top1 = main(ge)
        if top1 > best:
            best_ge = ge
            logger.info('Current best ge is:{0}'.format(str(best_ge)))
    logger.info('Final best ge is:{0}'.format(format(str(best_ge))))
