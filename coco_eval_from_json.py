from pycocotools.coco import COCO
from lib.util.coco_eval_oLRP_largerIoU import COCOeval
from lib.dataloader.dataloader import CocoDataset

if __name__ == '__main__':
    set_name = [iset for iset in 'val2017'.split('+')]
    dataset = CocoDataset('data/coco', set_name=set_name)
    
    # load results in COCO evaluation tool
    coco_true = dataset.coco
    coco_true = coco_true[list(coco_true.keys())[0]]
    coco_pred = coco_true.loadRes('./results/val2017_bbox_results.json')

    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = coco_pred.getImgIds()
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
