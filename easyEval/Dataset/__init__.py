from easyEval.Dataset.GSM8KDataset import GSM8KDataset
from easyEval.Dataset.MFQ30Dataset import MFQ30Dataset
from easyEval.Dataset.PVQRRDataset import PVQRRDataset

str2datasets = { "gsm8k": GSM8KDataset, "mfq30":MFQ30Dataset, "pvqrr": PVQRRDataset}



__all__ = ["str2datasets"]
