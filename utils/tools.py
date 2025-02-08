import numpy
import torch.distributed as dist
import torch
import clip
import os


def reduce_tensor(tensor, n=None):
    if n is None:
        n = dist.get_world_size()
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / n
    return rt
   

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def sync(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        val = torch.tensor(self.val).cuda()
        sum_v = torch.tensor(self.sum).cuda()
        count = torch.tensor(self.count).cuda()
        self.val = reduce_tensor(val, world_size).item()
        self.sum = reduce_tensor(sum_v, 1).item()
        self.count = reduce_tensor(count, 1).item()
        self.avg = self.sum / self.count


def epoch_saving(config, epoch, model,  max_accuracy, optimizer, lr_scheduler, logger, working_dir, is_best):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    
    save_path = os.path.join(working_dir, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
    if is_best:
        best_path = os.path.join(working_dir, f'best.pth')
        torch.save(save_state, best_path)
        logger.info(f"{best_path} saved !!!")


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    if os.path.isfile(config.MODEL.RESUME): 
        logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
        load_state_dict = checkpoint['model']

        msg = model.load_state_dict(load_state_dict, strict=False)
        logger.info(f"resume model: {msg}")

        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            start_epoch = checkpoint['epoch'] + 1
            max_accuracy = checkpoint['max_accuracy']

            logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
            
            del checkpoint
            torch.cuda.empty_cache()

            return start_epoch, max_accuracy
        except:
            del checkpoint
            torch.cuda.empty_cache()
            return 0, 0.

    else:
        logger.info(("=> no checkpoint found at '{}'".format(config.MODEL.RESUME)))
        return 0, 0


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file



def generate_text(data):
    prompt_templates = [
        '{}',
        'a video of {}.',
        'a video of a person {}.',
        #'a video of a person using {}.', # optional, Cancelling this part can aug training samples, but it will increase the time cost of validate
        #'a video of a person doing {}.',
        #'a video of a person during {}.',
        #'a video of a person performing {}.',
        #'a video of a person practicing {}.',
        'a example of {}.',
        'a example of a person {}.',
        #'a example of a person using {}.',
        #'a example of a person doing {}.',
        #'a example of a person during {}.',
        #'a example of a person performing {}.',
        #'a example of a person practicing {}.',
        'a demonstration of {}.',
        'a demonstration of a person {}.',
        #'a demonstration of a person using {}.',
        #'a demonstration of a person doing {}.',
        #'a demonstration of a person during {}.',
        #'a demonstration of a person performing {}.',
        #'a demonstration of a person practicing {}.',
        'Human action of {}.',
        '{}, an action.',
        '{} this is an action.',
        '{}, a video of action.',
        'Playing action of {}.',
        'Playing a kind of action, {}',
        'Doing a kind of action, {}',
        'Look, the human is {}',
        'Video classification of {}',
    ]
    classes, descriptions = data.classes
    C = len(classes)
    A = len(descriptions[0])
    P = len(prompt_templates)

    classes = torch.cat([clip.tokenize(prompt_templates[j].format(c), context_length=77) for j in range(P) for i, c in classes])
    classes = classes.reshape(P, C, 77)
    descriptions = torch.cat([clip.tokenize(descriptions[j][i], context_length=77) for i in range(A) for j in range(len(descriptions))])
    descriptions = descriptions.reshape(A, C, 77)
    classes = torch.cat([classes, descriptions], dim=0) # [P+A,C,content_length] simplify -> [A,C,content_length]
    return classes


if __name__ == '__main__':
    import pandas as pd
    '''Testing the generate_text function in tools.py'''
    def annotations(labels_file, descriptions_file):
        classes_all = pd.read_csv(labels_file)
        descriptions = {}
        if descriptions_file is not None:
            augment_texts0 = pd.read_csv(descriptions_file[0], sep=';')
            augment_texts1 = pd.read_csv(descriptions_file[1], sep=';')
            augment_texts2 = pd.read_csv(descriptions_file[2], sep=';')
            for i in range(len(classes_all)):
                descriptions[i] = augment_texts0[augment_texts0['id'] == i]['name'].tolist()
                descriptions[i] += augment_texts1[augment_texts1['id'] == i]['name'].tolist()
                descriptions[i] += augment_texts2[augment_texts2['id'] == i]['name'].tolist()
            return classes_all.values.tolist(), descriptions
        else:
            return classes_all.values.tolist()

    def generate_text(labels_file, descriptions_file):
        prompt_templates = [
            '{}',
            '{}',
            '{}',
            '{}',
            '{}',
        ]
        classes, descriptions = annotations(labels_file, descriptions_file)
        C = len(classes)
        A = len(descriptions[0])
        P = len(prompt_templates)
        classes = torch.cat(
            [clip.tokenize(prompt_templates[j].format(c), context_length=77) for j in range(P) for i, c in classes])

        classes = classes.reshape(P, C, 77)

        descriptions = torch.cat([clip.tokenize(descriptions[j][i], context_length=77) for i in range(A) for j in range(len(descriptions))])
        descriptions = descriptions.reshape(A, C, 77)
        classes = torch.cat([classes, descriptions], dim=0)  # [P+A,C,content_length]
        return classes

    labels_file = '/Users/yangjingyi/PycharmProjects/pythonProject/CLAVER_/labels/kinetics_400_labels.csv'
    descriptions_file = ['/Users/yangjingyi/PycharmProjects/pythonProject/CLAVER_/labels/kinetics_400_descriptions_ext.csv',
                         '/Users/yangjingyi/PycharmProjects/pythonProject/CLAVER_/labels/kinetics_400_body.csv',
                         '/Users/yangjingyi/PycharmProjects/pythonProject/CLAVER_/labels/kinetics_400_synonyms.csv']
    classes = generate_text(labels_file, descriptions_file)