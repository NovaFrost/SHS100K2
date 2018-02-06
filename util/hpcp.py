# Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
#
# This file is part of Essentia
#
# Essentia is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation (FSF), either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the Affero GNU General Public License
# version 3 along with this program. If not, see http://www.gnu.org/licenses/

#! /usr/bin/env python

import sys, os
import essentia, essentia.standard, essentia.streaming
from essentia.streaming import *
import numpy as np


tonalFrameSize = 4096
tonalHopSize = 1024

class TonalDescriptorsExtractor(essentia.streaming.CompositeBase):

    def __init__(self, frameSize=tonalFrameSize, hopSize=tonalHopSize, tuningFrequency=440.0):
        super(TonalDescriptorsExtractor, self).__init__()

        fc = FrameCutter(frameSize=frameSize,
                         hopSize=hopSize,
                         silentFrames='noise')

        w = Windowing(type='blackmanharris92')
        spec = Spectrum()
        peaks = SpectralPeaks(maxPeaks=30,
                              magnitudeThreshold=0.00001,
                              minFrequency=40,
                              maxFrequency=5000,
                              orderBy='magnitude');
        hpcp_key = HPCP(harmonics=7,
                        size = 12,
                        referenceFrequency = tuningFrequency,
                        bandPreset = False,
                        minFrequency = 40.0,
                        maxFrequency = 5000.0,
                        weightType = 'squaredCosine',
                        nonLinear = False,
                        windowSize = 4.0/3.0);

        fc.frame >> w.frame >> spec.frame
        spec.spectrum >> peaks.spectrum
        peaks.frequencies >> hpcp_key.frequencies
        peaks.magnitudes >> hpcp_key.magnitudes

        # define inputs:
        self.inputs['signal'] = fc.signal

        # define outputs:
        self.outputs['hpcp'] = hpcp_key.hpcp

usage = 'tonaldescriptors.py [options] <inputfilename> <outputfilename>'

def down_post(pool, v):
    hpcp = pool['hpcp']
    glo_hpcp = np.sum(hpcp, axis=0)
    glo_hpcp /= np.max(glo_hpcp) if np.max(glo_hpcp) > 0 else 1
    for i in xrange(0, len(hpcp) / v):
        down_item = np.sum(hpcp[i * v : (i + 1) * v, :], axis=0)
        down_item /= np.max(down_item) if np.max(down_item) > 0 else 1
        pool.add('down_sample_hpcp', down_item)
    pool.remove('hpcp')
    pool.set('glo_hpcp', glo_hpcp)

def parse_args():

    import numpy
    essentia_version = '%s\n'\
    'python version: %s\n'\
    'numpy version: %s' % (essentia.__version__,       # full version
                           sys.version.split()[0],     # python major version
                           numpy.__version__)          # numpy version

    from optparse import OptionParser
    parser = OptionParser(usage=usage, version=essentia_version)

    parser.add_option("-c","--cpp", action="store_true", dest="generate_cpp",
      help="generate cpp code from CompositeBase algorithm")

    parser.add_option("-d", "--dot", action="store_true", dest="generate_dot",
      help="generate dot and cpp code from CompositeBase algorithm")

    (options, args) = parser.parse_args()

    return options, args



if __name__ == '__main__':

    # sty: 1: hpcp_hpcp, 2: hpcp_npy, 4: 2dfm_npy
    opts, args = parse_args()
    in_path, out_path = args  
    file_name = in_path.split('/')[-1].split('.')[0]
    
    if opts.generate_dot:
        essentia.translate(TonalDescriptorsExtractor, 'streaming_extractortonaldescriptors', dot_graph=True)
    elif opts.generate_cpp:
        essentia.translate(TonalDescriptorsExtractor, 'streaming_extractortonaldescriptors', dot_graph=False)

    pool = essentia.Pool()
    loader = essentia.streaming.MonoLoader(filename=in_path)
    tonalExtractor = TonalDescriptorsExtractor()
    loader.audio >> tonalExtractor.signal
    for desc, output in tonalExtractor.outputs.items():
        output >> (pool, desc)
    essentia.run(loader)
    
    down_post(pool, 20)
    np.save(out_path, pool['down_sample_hpcp'])
    
