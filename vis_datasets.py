import base64
import math
import os
import re
from io import BytesIO

import scipy.io as sio
from IPython.display import display, HTML
from PIL import Image

"""
修改/home/luffy/.conda/envs/experiment/lib/python3.6/site-packages/notebook/static/custom/custom.css文件，使得输出表格的隔行颜色相同，
添加如下内容：

.rendered_html tbody tr:nth-child(odd) {
    background: #ffffff;
}

.rendered_html td {
    text-align: center;
}


刷新浏览器即可
"""


class Visualization(object):
    def __init__(self, row=1, col=1, image_width=50, image_height=50):
        self.title = ''
        self.header = ''
        self.row = row
        self.col = col
        self.max_col = 20
        self.figure_number = 0
        self.html_template = "<html><head>{css}</head><body><div>{html}</div></body><html>"
        self.figure_style = "<figure>"
        self.html = ''
        self.image_template = "<img style=\"object-fit:fill;padding:0px;margin:0px;width:{IMAGE_WIDTH}px;height:{IMAGE_HEIGHT}px\" src=\"{image_url}\"></img></figure>".replace(
            '{IMAGE_WIDTH}', str(image_width)).replace('{IMAGE_HEIGHT}', str(image_height))
        self.figure_caption_template = "<figcaption>{caption}</figcaption>"
        self.css = """
        <style type="text/css">
        figure {background-color:transparent; padding:0px;text-align:center;margin:0px;object-fit:fill;}
        table.center_table {font-size:9px;background-color:transparent;border-collapse:collapse; text-align:center}
        table.center_table td {border:0px solid black;padding:0px;margin:0px;width:100%;}
        table.center_table td {width:{IMAGE_WIDTH}px; height:{IMAGE_HEIGHT}px;}
        </style>
        """.replace('{IMAGE_WIDTH}', str(image_width)).replace('{IMAGE_HEIGHT}', str(image_height))

    def row_label(self, r):
        return 'row_{}'.format(r)

    def col_label(self, c):
        return 'col_{}'.format(c)

    def format_row_col(self, r, c):
        return '{%s_%s}' % (self.row_label(r), self.col_label(c))

    def empty_lefttop(self):
        self.html = self.fill_cell(0, 0, "")

    def empty_leftbottom(self):
        self.html = self.fill_cell(self.row - 1, 0, "")

    def fill_cell(self, r, c, data):
        index = self.format_row_col(r, c)
        self.html = self.html.replace(index, data)
        return self.html

    def make_table(self):
        html = '<table class="center_table">'
        # add header
        html += '<tr align="center"><td colspan="%d"><b>%s</b></td></tr>' % (self.col, self.header)
        for i in range(self.row):
            html += '<tr>'
            for j in range(self.col):
                html += '<td>{index}</td>'.format(index=self.format_row_col(i, j))
            html += '</tr>'
        html += '</table>'
        return html

    def format_html(self, css, html):
        return self.html_template.format(css=css, html=html)

    def format_image(self, url):
        return self.image_template.format(image_url=url)

    def clean_caption(self, caption):
        return caption

    def format_caption(self, caption):
        return self.figure_caption_template.format(caption=caption)

    def add_image(self, image_url, row, col, caption, no_image=False):
        if not self.html:
            self.html = self.make_table()

        def prepare(image_url, caption):
            caption = self.clean_caption(caption)
            if no_image:
                html = '<b>%s</b>' % caption.replace('_', '<br>')
            else:
                html = self.figure_style
                html += self.format_caption(caption)
                html += self.format_image(image_url)
            return html

        data = prepare(image_url, caption)
        self.html = self.fill_cell(row, col, data)

    def add_all(self):
        pass

    def _show(self):
        display(HTML(self.format_html(self.css, self.html)))

    def show(self):
        self.add_all()
        self._show()


class DomainNet(Visualization):
    def __init__(self, base_dir):
        super(DomainNet, self).__init__(row=1, col=1)
        self.base_dir = base_dir
        self.domains = []
        self.classes = []
        # update header
        self.header = 'DomainNet'

    def get_domains(self):
        dirs = os.listdir(self.base_dir)
        # filter
        dirs = list(filter(lambda x: os.path.isdir(os.path.join(self.base_dir, x)), dirs))
        return sorted(dirs)

    def get_classes(self, domains):
        classes = []
        if not domains:
            return classes
        classes = os.listdir(os.path.join(self.base_dir, domains[0], domains[0]))
        return sorted(classes, key=lambda x: x.lower())

    def get_first_image(self, domain, _class):
        files = os.listdir(os.path.join(self.base_dir, domain, domain, _class))
        files = sorted(files)
        first = os.path.join(self.base_dir, domain, domain, _class, files[0])
        return first

    def add_all(self):
        # update
        domains = self.get_domains() if not self.domains else self.domains
        classes = self.get_classes(domains) if not self.classes else self.classes

        # self.row and self.col are used to make table
        # row = len(domains)
        self.row = len(domains) + 1
        # first col shows domains
        self.col = self.cal_col_count(classes)
        for r, domain in enumerate(domains, start=0):
            # domains
            self.add_image('', r, 0, caption=domain, no_image=True)

            # images
            for c, _class in enumerate(classes, start=1):
                first_image = self.get_first_image(domain, _class)
                self.add_image(first_image, r, c, caption='')

            # fill empty
            for idx in range(len(classes), self.col):
                self.fill_cell(r, idx, "")

        for c, _class in enumerate(classes, start=1):
            self.add_image('', len(domains), c, caption=_class, no_image=True)
        # fill empty
        for idx in range(len(classes), self.col):
            self.fill_cell(len(domains), idx, "")
        # clear left bottom
        self.empty_leftbottom()

    def cal_col_count(self, classes):
        return self.max_col if len(classes) + 1 > self.max_col else len(classes) + 1

    @classmethod
    def show_all(cls, path):
        dn = cls(path)
        # all domains and all classes
        all_domains = dn.get_domains()

        if len(all_domains) == 1:
            # without domain label
            all_classes = dn.get_classes(all_domains)
            col_count = dn.cal_col_count(all_classes)
            # cal row count
            row_count = int(math.ceil(len(all_classes) / col_count))
            dn.row = row_count * 2
            dn.col = col_count
            for r in range(row_count):
                for c in range(col_count):
                    if r * col_count + c >= len(all_classes):
                        # fill empty
                        dn.fill_cell(2 * r, c, "")
                        dn.fill_cell(2 * r + 1, c, "")
                    else:
                        _class = all_classes[r * col_count + c]
                        first_image = dn.get_first_image(all_domains[0], _class)
                        dn.add_image(first_image, 2 * r, c, caption='')
                        dn.add_image('', 2 * r + 1, c, caption=_class, no_image=True)
            dn._show()

        else:
            all_classes = dn.get_classes(all_domains)
            # real col count
            col_count = dn.cal_col_count(all_classes) - 1  # first col is domains
            # loop to show all columns
            figure_count = int(math.ceil(len(all_classes) / col_count))

            for idx in range(figure_count):
                dni = cls(path)
                dni.figure_number = idx
                dni.classes = all_classes[idx * col_count: (idx + 1) * col_count]
                # update header
                if figure_count == 1:
                    dni.header = '{title} Samples Example'.format(title=dni.header)
                else:
                    dni.header = '{title} Samples Example {i}/{count}'.format(title=dni.header, i=idx + 1,
                                                                              count=figure_count)
                dni.add_all()
                dni.show()


class Office31(DomainNet):
    def __init__(self, base_dir):
        super(Office31, self).__init__(base_dir)
        self.header = 'Office 31'

    def get_classes(self, domains):
        classes = []
        if not domains:
            return classes
        classes = os.listdir(os.path.join(self.base_dir, domains[0], 'images'))
        return sorted(classes, key=lambda x: x.lower())

    def get_first_image(self, domain, _class):
        files = os.listdir(os.path.join(self.base_dir, domain, 'images', _class))
        files = sorted(files)
        first = os.path.join(self.base_dir, domain, 'images', _class, files[0])
        return first


class COIL20(DomainNet):
    def __init__(self, base_dir):
        super(COIL20, self).__init__(base_dir)
        self.header = 'COIL 20'
        self.max_col = 5

    def get_domains(self):
        return ['obj']

    def get_classes(self, domains):
        return list(map(str, range(1, 21)))

    def get_first_image(self, domain, _class):
        filename = '{d}{c}__0.png'.format(d=domain, c=_class)
        first = os.path.join(self.base_dir, filename)
        return first


class ImageCLEF(DomainNet):
    def __init__(self, base_dir):
        super(ImageCLEF, self).__init__(base_dir)
        self.header = 'ImageCLEF'
        self.max_col = 13
        self.cache = {}

    def get_domains(self):
        return ['Caltech', 'ImageNet', 'Pascal']

    def get_classes(self, domains):
        _classes = """
        aeroplane
        bike
        bird
        boat
        bottle
        bus
        car
        dog
        horse
        monitor
        motorbike
        people
        """
        return list(filter(lambda x: x, map(lambda x: x.strip(), _classes.split('\n'))))

    def get_first_image(self, domain, _class):

        def process_listfile(domain):
            filename = os.path.join(self.base_dir, 'list', domain[0].lower() + 'List.txt')
            idx_filename_dict = {}
            with open(filename, 'r') as f:
                while f.readable():
                    line = f.readline()
                    if not line.strip():
                        break
                    name = line.split(' ')[0]
                    # replace
                    name = re.sub('/[^/]+/', '/', name)
                    class_idx = line.split(' ')[1].strip()
                    if str(class_idx) not in idx_filename_dict:
                        idx_filename_dict[str(class_idx)] = name
            return idx_filename_dict

        # use cache
        if domain not in self.cache:
            self.cache[domain] = process_listfile(domain)
        cached = self.cache[domain]
        class_idx = self.get_classes([]).index(_class)
        filename = cached.get(str(class_idx))
        first = os.path.join(self.base_dir, filename)
        return first


class MnistUsps(DomainNet):
    def __init__(self, base_dir):
        super(MnistUsps, self).__init__(base_dir)
        self.header = 'MNIST-USPS'
        self.max_col = 11
        self.cache = {}

    def get_domains(self):
        return ['MNIST', 'USPS']

    def get_classes(self, domains):
        return list(map(str, range(10)))

    def image_2_base64(self, image_data):
        img = Image.fromarray(image_data)
        output_buffer = BytesIO()
        img.save(output_buffer, format='JPEG')
        im_data = output_buffer.getvalue()
        image_data = base64.b64encode(im_data)
        if not isinstance(image_data, str):
            # Python 3, decode from bytes to string
            image_data = image_data.decode()
        data_url = 'data:image/jpg;base64,' + image_data
        return data_url

    def get_first_image(self, domain, _class):
        def read_mat(filename):
            return sio.loadmat(filename)

        def get_filename(domain):
            return os.path.join(self.base_dir, '%s_all.mat' % domain.lower())

        if domain not in self.cache:
            self.cache[domain] = read_mat(get_filename(domain))

        data = self.cache[domain]
        if domain.upper() == 'MNIST':
            image_data = data['train' + _class][0, :].reshape((28, 28))
        else:
            image_data = data['data'][:, 0, int(_class)].reshape((16, 16))

        data = self.image_2_base64(image_data)
        return data


class OfficeCaltech(DomainNet):
    def __init__(self, base_dir):
        super(OfficeCaltech, self).__init__(base_dir)
        self.header = 'Office Caltech'
        self.max_col = 11

    def get_classes(self, domains):
        classes = []
        if not domains:
            return classes
        classes = os.listdir(os.path.join(self.base_dir, domains[0]))
        # format class names
        classes = list(map(lambda x: x.replace('backpack', 'back_pack'), classes))
        return sorted(classes, key=lambda x: x.lower())

    def get_first_image(self, domain, _class):
        # fixed name
        if domain.lower() == 'caltech' and _class.lower() == 'back_pack':
            _class = 'backpack'
        files = os.listdir(os.path.join(self.base_dir, domain, _class))
        files = sorted(files)
        first = os.path.join(self.base_dir, domain, _class, files[0])
        return first


class OfficeHome(DomainNet):
    def __init__(self, base_dir):
        super(OfficeHome, self).__init__(base_dir)
        self.header = 'Office Home'

    def get_classes(self, domains):
        classes = []
        if not domains:
            return classes
        classes = os.listdir(os.path.join(self.base_dir, domains[0]))
        return sorted(classes, key=lambda x: x.lower())

    def get_first_image(self, domain, _class):
        files = os.listdir(os.path.join(self.base_dir, domain, _class))
        files = sorted(files)
        first = os.path.join(self.base_dir, domain, _class, files[0])
        return first


class VisDA(OfficeHome):
    def __init__(self, base_dir):
        super(VisDA, self).__init__(base_dir)
        self.header = 'VisDA'

    def get_classes(self, domains):
        classes = super(VisDA, self).get_classes(domains)
        # filter
        classes = list(filter(lambda x: not x.endswith('.txt'), classes))
        return classes


class VLSC(DomainNet):
    """
    VOC2007
    LabelMe
    Sun09
    Caltech

    'bird', 'car', 'chair', 'dog', 'person'
    """

    def __init__(self, base_dir):
        super(VLSC, self).__init__(base_dir)
        self.header = 'VLSC'
        self.domain_class_image_dict = {}

    def get_domains(self):
        return ['VOC2007', 'LabelMe', 'SUN09', 'Caltech']

    def get_classes(self, domains):
        return ['bird', 'car', 'chair', 'dog', 'person']

    @property
    def voc2007(self):
        return os.path.join(self.base_dir, 'VOCdevkit/VOC2007')

    @property
    def labelme(self):
        return os.path.join(self.base_dir, 'LabelMe')

    @property
    def sun09(self):
        return os.path.join(self.base_dir, 'sun09')

    @property
    def caltech(self):
        return os.path.join(self.base_dir, 'Caltech')

    def process_voc2007(self):
        class_first_image_dict = {}
        base = self.voc2007
        txt_folder = os.path.join(base, 'ImageSets/Main')
        filename_template = '{0}_train.txt'
        _classes = self.get_classes([])
        for c in _classes:
            filename = os.path.join(txt_folder, filename_template.format(c))
            # read file
            with open(filename, 'r') as f:
                while f.readable():
                    line = f.readline().strip()
                    if not line:
                        break
                    line = line.replace('  ', ' ')
                    img_name, class_idx = line.split(' ')
                    if class_idx == '1' and c not in class_first_image_dict:
                        class_first_image_dict[c] = os.path.join(base, 'JPEGImages', '%s.jpg' % img_name)
                        break
        return class_first_image_dict

    def process_labelme(self):

        class_first_image_dict = {}
        _classes = self.get_classes([])
        base = self.labelme
        # get first image
        for c in _classes:
            folder = sorted(os.listdir(os.path.join(base, c, 'images')))[0]
            filename = sorted(os.listdir(os.path.join(base, c, 'images', folder)))[0]
            class_first_image_dict[c] = os.path.join(base, c, 'images', folder, filename)
        return class_first_image_dict

    def process_sun09(self):
        class_first_image_dict = {}
        _classes = self.get_classes([])
        base = self.sun09
        # get first image
        for c in _classes:
            filename = sorted(os.listdir(os.path.join(base, c)))[0]
            class_first_image_dict[c] = os.path.join(base, c, filename)
        return class_first_image_dict

    def process_caltech(self):
        name_map = {
            'bird': '113.hummingbird',
            'person': '159.people',
            'dog': '056.dog',
            'car': 'car_side',
            # 'chair': 'chair'
        }
        class_first_image_dict = {}
        base = self.caltech
        _classes = self.get_classes([])
        for c in _classes:
            name = name_map.get(c, c)
            filename = sorted(os.listdir(os.path.join(base, name)))[0]
            class_first_image_dict[c] = os.path.join(base, name, filename)
        return class_first_image_dict

    def get_first_image(self, domain, _class):
        func_map = {
            'VOC2007': self.process_voc2007,
            'LabelMe': self.process_labelme,
            'SUN09': self.process_sun09,
            'Caltech': self.process_caltech,
        }
        if domain not in self.domain_class_image_dict:
            self.domain_class_image_dict[domain] = func_map.get(domain)()
        return self.domain_class_image_dict.get(domain, {}).get(_class)


class CrossDataset(OfficeHome):
    def __init__(self, base_dir):
        super(CrossDataset, self).__init__(base_dir)
        self.header = 'CrossDataset'
        self.max_col = 21

    def get_domains(self):
        return ['Caltech', 'Imagenet', 'SUN']

    def get_classes(self, domains):
        classes = os.listdir(os.path.join(self.base_dir, 'imagenet'))
        classes = list(filter(lambda x: os.path.isdir(os.path.join(self.base_dir, 'imagenet', x)), classes))
        return sorted(classes)

    def get_first_image(self, domain, _class):
        domain = domain.lower()
        class_idx = (self.max_col - 1) * self.figure_number + self.classes.index(_class) + 1
        if domain in ('caltech', 'sun'):
            folder = os.path.join(self.base_dir, domain, str(class_idx))
        else:
            folder = os.path.join(self.base_dir, domain, _class)

        # sorted
        first_image = sorted(os.listdir(folder))[0]
        first_image = os.path.join(folder, first_image)
        return first_image


class PIE(OfficeHome):
    def __init__(self, base_dir):
        super(PIE, self).__init__(base_dir)
        self.header = 'PIE'
        self.max_col = 21
        self.folder_tempate = 'Pose{domain}_64x64_files'

    def get_domains(self):
        return ['05', '07', '09', '27', '29']

    def get_classes(self, domains):
        return list(map(str, range(1, 21)))

    def get_first_image(self, domain, _class):
        filename = os.path.join(self.base_dir, self.folder_tempate.format(domain=domain), '%s.jpg' % _class)
        return filename


class VOC2012(OfficeHome):
    def __init__(self, base_dir):
        super(VOC2012, self).__init__(base_dir)
        self.header = 'PASCAL VOC2012'
        self.max_col = 5

    def get_domains(self):
        return ['']

    def get_classes(self, domains):
        folders = os.listdir(os.path.join(self.base_dir))
        classes = sorted(list(filter(lambda x: os.path.isdir(os.path.join(self.base_dir, x)), folders)))
        return classes

    def get_first_image(self, domain, _class):
        files = os.listdir(os.path.join(self.base_dir, _class))
        files = sorted(files)
        first = os.path.join(self.base_dir, _class, files[0])
        return first


class StanfordDogs(OfficeHome):
    def __init__(self, base_dir):
        super(StanfordDogs, self).__init__(base_dir)
        self.header = 'Stanford Dogs'
        self.max_col = 15
        self.name_remap = {
            'curly-coated_retriever': 'curly-<br>coated_retriever',
            'affenpinscher': 'affenpin-<br>scher',
            'Chihuahua': 'Chihua-<br>hua',
            'EntleBucher': 'Entle-<br>Bucher',
            'Pomeranian': 'Pomera-<br>nian',
            'Staffordshire_bullterrier': 'Stafford-<br>shire_bullterr-<br>ier',
            'bloodhound': 'blood-<br>hound',
            'Weimaraner': 'Weimaran-<br>er',
            'Chesapeake_Bay_retriever': 'Chesap-<br>eake_Bay_retriever',
            'groenendael': 'groenen-<br>dael',
            'black-and-tan_coonhound': 'black-<br>and-<br>tan_coonhound',
            'American_Staffordshire_terrier': 'American_Stafford-<br>shire_terrier',
            'German_short-haired_pointer': 'German_short-<br>haired_pointer',
            'Appenzeller': 'Appenzell-<br>er',
            'Newfoundland': 'Newfound-<br>land',
            'otterhound': 'otter-<br>hound',
            'schipperke': 'schip-<br>perke',
            'Brabancon_griffon': 'Braban-<br>con_griffon',
        }

    def get_domains(self):
        return ['']

    def get_classes(self, domains):
        folders = os.listdir(os.path.join(self.base_dir))
        classes = sorted(list(filter(lambda x: os.path.isdir(os.path.join(self.base_dir, x)), folders)))
        return classes

    def get_first_image(self, domain, _class):
        files = os.listdir(os.path.join(self.base_dir, _class))
        files = sorted(files)
        first = os.path.join(self.base_dir, _class, files[0])
        return first

    def clean_caption(self, caption):
        if '-' in caption:
            caption = caption[caption.index('-') + 1:]
        caption = self.name_remap.get(caption, caption)
        return caption


class IntelImageClassification(OfficeHome):
    def __init__(self, base_dir):
        super(IntelImageClassification, self).__init__(base_dir)
        self.header = 'Intel Image Classification'
        self.max_col = 6

    def get_domains(self):
        return ['']

    def get_classes(self, domains):
        folders = os.listdir(os.path.join(self.base_dir))
        classes = sorted(list(filter(lambda x: os.path.isdir(os.path.join(self.base_dir, x)), folders)))
        return classes

    def get_first_image(self, domain, _class):
        files = os.listdir(os.path.join(self.base_dir, _class))
        files = sorted(files)
        first = os.path.join(self.base_dir, _class, files[0])
        return first


if __name__ == '__main__':
    """
    Amazon review dataset
    """
    # vis = Visualization(row=3, col=2)
    # # vis.show_image()
    # image_url = 'dataset/DomainNet/clipart/clipart/whale/clipart_337_000342.jpg'
    # vis.add_image(image_url, 0, 0, caption='test')
    # vis.add_image(image_url, 0, 1, caption='tes2')
    # vis.add_image(image_url, 1, 0, caption='')
    # vis.add_image(image_url, 1, 1, caption='')
    # vis.add_image(image_url, 2, 0, caption='')
    # vis.add_image(image_url, 2, 1, caption='')
    # vis.show()

    # dn = DomainNet('dataset/DomainNet')
    # dn.add_all()
    # dn.show()
    # DomainNet.show_all('dataset/DomainNet')
    # Office31.show_all('dataset/office31_raw_image/Original_images')
    # COIL20.show_all('dataset/for_vis/coil-20-proc')
    # ImageCLEF.show_all('dataset/for_vis/image_CLEF')
    # MnistUsps.show_all('dataset/for_vis/mnist+usps')
    # OfficeCaltech.show_all('dataset/for_vis/office_caltech_10')
    # OfficeHome.show_all('dataset/for_vis/OfficeHomeDataset_10072016')
    # VisDA.show_all('dataset/for_vis/visda')
    # VLSC.show_all('dataset/for_vis/vlsc')
    # CrossDataset.show_all('dataset/for_vis/crossdataset')
    # PIE.show_all('dataset/for_vis/PIE/PIE')
    # VOC2012.show_all('dataset/VOCdevkit/cleaned/train')
    StanfordDogs.show_all('graduation/dataset/Stanford_Dogs/Images')
    # IntelImageClassification.show_all('dataset/intel-image-classification/cleaned/train')
