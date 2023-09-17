import yaml
from os.path import join

from torch.utils.data import Dataset

from utils.types import *

from datasets.cadc import CADCProxySequence


class CADCDatasetLidarOnly(Dataset):

    def __init__(self,
                 base_path: str,
                 day2sequences: Dict[str, List[str]]
                 ):

        self.path = base_path
        self.sequences = []
        for day, seqs in day2sequences.items():
            for seq in seqs:
                self.sequences.append(CADCProxySequence(join(base_path, day, seq),
                                                        join(base_path, day, 'calib')))

        self.idx2frame = []

        for seq_idx in range(len(self.sequences)):
            for frame_idx in range(len(self.sequences[seq_idx])):
                self.idx2frame.append({
                    'seq_idx': seq_idx, 'frame_idx': frame_idx
                })

    def __len__(self) -> int:
        return sum([len(seq) for seq in self.sequences])

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Dict[str, yaml.YAMLObject]]:

        info = self.idx2frame[idx]
        frame = self.sequences[info['seq_idx']][info['frame_idx']]
        calib = self.sequences[info['seq_idx']].get_calib()

        return Tensor(frame['points']), Tensor(frame['boxes']), calib
