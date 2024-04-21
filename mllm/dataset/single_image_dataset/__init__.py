from .flickr import FlickrParser, FlickrDataset
from .rec import RECDataset, RECComputeMetrics
from .caption import CaptionDataset
from .instr import InstructDataset
from .gqa import GQADataset, GQAComputeMetrics
from .clevr import ClevrDataset
from .point_qa import Point_QA_local, Point_QA_twice, V7W_POINT, PointQAComputeMetrics
from .gpt_gen import GPT4Gen
from .gpt_gen_mask import GPT4GenMask
from .vcr import VCRDataset, VCRPredDataset
from .vqav2 import VQAv2Dataset
from .vqaex import VQAEXDataset
from .pope import POPEVQADataset
from .ref_mask import REFMaskDataset, RECMaskComputeMetrics
from .instance_seg import InstanceSegDataset
from .box2seg import Box2SegDataset
from .gref import Gref
from .ref_mask_vg import REFMaskVGDataset
from .flickr_mask import FlickrMaskDataset
from .ref_mask_refcoco import REFMaskRefcocoDataset
from .reg import REGDataset, GCDataset
from .reg_mask import GCMaskDataset
