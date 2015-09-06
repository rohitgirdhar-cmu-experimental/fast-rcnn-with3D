# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.pascal_voc
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess

class sunrgbd(datasets.imdb):
    def __init__(self, image_set, data_path):
        datasets.imdb.__init__(self, 'sunrgbd_' + image_set)
        self._image_set = image_set # train/test/val
        self._data_path = data_path
        self._classes = ('__background__', # always index 0
                         'bathtub', 'bed', 'bookshelf', 'box',
                         'chair', 'counter', 'desk', 'door',
                         'dresser', 'garbage_bin', 'lamp',
                         'monitor', 'night_stand', 'pillow', 'sink',
                         'sofa', 'table', 'tv', 'toilet')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '' # the image index includes this information
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        # SUNRGBD specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

#        assert os.path.exists(self._devkit_path), \
#                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt

        image_sets_file = os.path.join(self._data_path, 'Toolbox', 
                                      'SUNRGBDtoolbox', 'traintestSUNRGBD',
                                      'allsplit.mat')
        assert os.path.exists(image_sets_file), \
                'Path does not exist: {}'.format(image_sets_file)
        image_sets = sio.loadmat(image_sets_file, struct_as_record=False, squeeze_me=True)
        if self._image_set == 'train':
          seq_names = image_sets['trainvalsplit'].train
        elif self._image_set == 'val':
          seq_names = image_sets['trainvalsplit'].val
        elif self._image_set == 'trainval':
          seq_names = image_sets['alltrain']
        elif self._image_set == 'test':
          seq_names = image_sets['alltest']
        else:
          sys.stderr.write('Incorrect imageset!\n')
        # get the image sequence names from these image_index
        image_index = []
        for i in range(len(seq_names)):
          image_index.append(seq_names[i][len('/n/fs/sun3d/data/'):].rstrip('/'))
        
        final_image_index = []
        corr_image_ids = []
        all_imgs_list = sio.loadmat(os.path.join(self._data_path, 
              'Toolbox/SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat'), 
              struct_as_record=False)
        for i in range(len(all_imgs_list['SUNRGBDMeta'][0])):
          seq_name = all_imgs_list['SUNRGBDMeta'][0][i].sequenceName[0]
          if seq_name in image_index:
            final_image_index.append(os.path.join(seq_name, 
                'image', 
                all_imgs_list['SUNRGBDMeta'][0][i].rgbname[0]))
            corr_image_ids.append(i+1)
        self._image_index_ids = corr_image_ids
        return final_image_index

    def _get_default_path(self):
        """
        Return the default path where SUNRGBD is expected to be installed.
        """
        return self._data_path

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self._load_sunrgbd_annotations()
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_roidb(gt_roidb)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def selective_search_IJCV_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                '{:s}_selective_search_IJCV_top_{:d}_roidb.pkl'.
                format(self.name, self.config['top_k']))

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        import pdb
        pdb.set_trace()
        ss_roidb = self._load_selective_search_IJCV_roidb(gt_roidb)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_IJCV_roidb(self, gt_roidb):
        IJCV_path = os.path.abspath(os.path.join(self.cache_path, '..',
                                                 'selective_search_IJCV_data',
                                                 'voc_' + self._year))
        assert os.path.exists(IJCV_path), \
               'Selective search IJCV data not found at: {}'.format(IJCV_path)

        top_k = self.config['top_k']
        box_list = []
        for i in xrange(self.num_images):
            filename = os.path.join(IJCV_path, self.image_index[i] + '.mat')
            raw_data = sio.loadmat(filename)
            box_list.append((raw_data['boxes'][:top_k, :]-1).astype(np.uint16))

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _read_sunrgbd_annot_file(self):
        """
        Load image and bounding boxes info from MAT file in the SUNGRGBD
        format, to a dict(seq_name => info)
        """
        filename = os.path.join(self._data_path, 
            'Toolbox/SUNRGBDtoolbox/Metadata/groundtruth.mat')
        gtstruct = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)

        res = {}
        for i in range(len(gtstruct['groundtruth'])):
          sqname = gtstruct['groundtruth'][i].sequenceName.rstrip('/')
          if sqname in res.keys():
            res[sqname].append((gtstruct['groundtruth'][i].classname, 
                  gtstruct['groundtruth'][i].gtBb2D))
          else:
            res[sqname] = [(gtstruct['groundtruth'][i].classname, 
                  gtstruct['groundtruth'][i].gtBb2D)]
        return res

    def _load_sunrgbd_annotations(self):
        """
        Load image and bounding boxes info from MAT file in the SUNGRGBD
        format.
        """
        res = []
        all_annots = self._read_sunrgbd_annot_file()
        for imidx in self.image_index:        
          annots = []
          if imidx in all_annots.keys():
            annots = all_annots[imidx]
          
          num_objs = len(annots)
          boxes = np.zeros((num_objs, 4), dtype=np.uint16)
          gt_classes = np.zeros((num_objs), dtype=np.int32)
          overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
          
          for ix, annot in enumerate(annots):
            x1 = annot[1][0] - 1
            y1 = annot[1][1] - 1
            x2 = annot[1][2] - 1
            y2 = annot[1][3] - 1
            cls = self._class_to_ind[annot[0].lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
          overlaps = scipy.sparse.csr_matrix(overlaps)
          res.append({'boxes' : boxes,
                      'gt_classes' : gt_classes,
                      'gt_overlaps' : overlaps,
                      'flipped' : False})
        return res

    def _write_voc_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())

        # VOCdevkit/results/VOC2007/Main/comp4-44503_det_test_aeroplane.txt
        path = os.path.join(self._devkit_path, 'results', 'VOC' + self._year,
                            'Main', comp_id + '_')
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = path + 'det_' + self._image_set + '_' + cls + '.txt'
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        return comp_id

    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._devkit_path, comp_id,
                       self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        comp_id = self._write_voc_results_file(all_boxes)
        self._do_matlab_eval(comp_id, output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed; embed()
