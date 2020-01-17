from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--which_ref', type=str, default='path/to/reference/image', help='path to customized reference image')
        self.parser.add_argument('--csv_path', type=str, default='path/to/csv/driving', help='path to csv file containing AUs of driving video')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        self.parser.add_argument('--which_ref_d1', type=str, default='path/to/reference/image', help='path to customized driving_1 image')
        self.parser.add_argument('--which_ref_d2', type=str, default='path/to/reference/image', help='path to customized driving_2 image')
        self.isTrain = False
