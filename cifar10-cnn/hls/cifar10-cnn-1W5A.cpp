//*************************************************************************
// Copyright (C) 2018 Kaan Kara - Systems Group, ETH Zurich

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.

// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.
//*************************************************************************


#define TESTBENCH
// #define DEBUG

#define AP_INT_MAX_W 16384

#include "hls-nn-lib.h"
#include "../training/cifar10-cnn-config-1W5A.h"
#include "../training/cifar10-cnn-params-1W5A.h"

void DoCompute(stream<ap_axis >& in, stream<ap_axis >& out, const unsigned int numReps) {
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE s_axilite port=numReps bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS RESOURCE variable=weights0 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights0 complete dim=0
#pragma HLS RESOURCE variable=factorA0 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB0 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=factorA0 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB0 complete dim=0
#pragma HLS RESOURCE variable=weights1 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights1 complete dim=0
#pragma HLS RESOURCE variable=factorA1 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB1 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=factorA1 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB1 complete dim=0
#pragma HLS RESOURCE variable=weights2 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights2 complete dim=0
#pragma HLS RESOURCE variable=factorA2 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB2 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=factorA2 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB2 complete dim=0
#pragma HLS RESOURCE variable=weights3 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights3 complete dim=0
#pragma HLS RESOURCE variable=factorA3 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB3 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=factorA3 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB3 complete dim=0
#pragma HLS RESOURCE variable=weights4 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights4 complete dim=0
#pragma HLS RESOURCE variable=factorA4 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB4 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=factorA4 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB4 complete dim=0


#pragma HLS DATAFLOW
    // height 32, width 32
    // pixel_bits = 8*3 -- 8bit and 3 channels
    // pixels_per_line = 384/piexl_bits = 16
	// num lines = int((height*width)/pixels_per_line = 32*32/16 == 32*2 = 64 
	const unsigned int NumLinesPerRep = 64;

	stream<ap_uint<384> > in_stream_extract("in_stream_extract");
	ExtractPixels<384, NumLinesPerRep> (in, in_stream_extract, numReps);

	stream<ap_uint<L0_Cin*L0_Ibit> > in_stream("in_stream");
	ReduceWidth<384, L0_Cin*L0_Ibit, NumLinesPerRep> (in_stream_extract, in_stream, numReps);

#ifdef DEBUG
	Monitor<L0_Din, L0_Cin, L0_Ibit>(in_stream, (char*)"./log/mon_in_stream.log", numReps);
#endif

stream<ap_uint<L0_Cout*L0_Abit> > conv0("conv0");
CONV2D_ACT_NoP<L0_K, L0_S, L0_Din, L0_Cin, L0_Cout, L0_Ibit, L0_Wbit, L0_Mbit, L0_Abit, L0_MVTU_InP, L0_MVTU_OutP, SCALE_BITS, FACTOR_SCALE_BITS>
(in_stream, weights0, factorA0, factorB0, conv0, numReps);
#ifdef DEBUG
Monitor<L0_Din/L0_S, L0_Cout, L0_Abit>(conv0, (char*)"log/mon_conv0.log", numReps);
#endif

stream<ap_uint<L5_Cin*L5_Ibit> > pool0("pool0");
POOL2D_NoP<L5_K, L5_S, L5_Din, L5_Cin, L5_Ibit>
(conv0, pool0, numReps);
#ifdef DEBUG
Monitor<L5_Din/L5_S, L5_Cin, L5_Ibit>(pool0, (char*)"log/mon_pool0.log", numReps);
#endif

stream<ap_uint<L1_Cout*L1_Abit> > conv1("conv1");
CONV2D_ACT_NoP<L1_K, L1_S, L1_Din, L1_Cin, L1_Cout, L1_Ibit, L1_Wbit, L1_Mbit, L1_Abit, L1_MVTU_InP, L1_MVTU_OutP, SCALE_BITS, FACTOR_SCALE_BITS>
(pool0, weights1, factorA1, factorB1, conv1, numReps);
#ifdef DEBUG
Monitor<L1_Din/L1_S, L1_Cout, L1_Abit>(conv1, (char*)"log/mon_conv1.log", numReps);
#endif

stream<ap_uint<L2_Cout*L2_Abit> > conv2("conv2");
CONV2D_ACT_NoP<L2_K, L2_S, L2_Din, L2_Cin, L2_Cout, L2_Ibit, L2_Wbit, L2_Mbit, L2_Abit, L2_MVTU_InP, L2_MVTU_OutP, SCALE_BITS, FACTOR_SCALE_BITS>
(conv1, weights2, factorA2, factorB2, conv2, numReps);
#ifdef DEBUG
Monitor<L2_Din/L2_S, L2_Cout, L2_Abit>(conv2, (char*)"log/mon_conv2.log", numReps);
#endif

stream<ap_uint<L6_Cin*L6_Ibit> > pool1("pool1");
POOL2D_NoP<L6_K, L6_S, L6_Din, L6_Cin, L6_Ibit>
(conv2, pool1, numReps);
#ifdef DEBUG
Monitor<L6_Din/L6_S, L6_Cin, L6_Ibit>(pool1, (char*)"log/mon_pool1.log", numReps);
#endif

// stream<ap_uint<L3_Cout*L3_Abit> > conv3("conv3");
// CONV2D_ACT_NoP<L3_K, L3_S, L3_Din, L3_Cin, L3_Cout, L3_Ibit, L3_Wbit, L3_Mbit, L3_Abit, L3_MVTU_InP, L3_MVTU_OutP, SCALE_BITS, FACTOR_SCALE_BITS>
// (pool1, weights3, factorA3, factorB3, conv3, numReps);
// #ifdef DEBUG
// Monitor<L3_Din/L3_S, L3_Cout, L3_Abit>(conv3, (char*)"log/mon_conv3.log", numReps);
// #endif

stream<ap_uint<L3_OutP*L3_Abit> > dense0("dense0");
DENSE_ACT<L3_Din, L3_Dout, L3_Ibit, L3_Wbit, L3_Mbit, L3_Abit, L3_InP, L3_OutP, SCALE_BITS, FACTOR_SCALE_BITS>
(pool1, weights3, factorA3, factorB3, dense0, numReps);

stream<ap_uint<L4_OutP*L4_Mbit> > dense1("dense1");
DENSE_NOACT<L4_Din, L4_Dout, L4_Ibit, L4_Wbit, L4_Mbit, L4_InP, L4_OutP, SCALE_BITS>
(dense0, weights4, dense1, numReps);

	stream<ap_uint<512> > out_nolast("out_nolast");
	AppendZeros<10*L4_Mbit, 512, 1> (dense1, out_nolast, numReps);

	AddLast<1>(out_nolast, out, numReps);
}

// TESTBENCH
#ifdef TESTBENCH
#include <iostream>
#include <fstream>
#include <string>

#ifdef REAL_IMAGE
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
#endif

using namespace std;

int main(int argc, char* argv[]) {
    const unsigned NUM_SAMPLES=1;


#ifdef REAL_IMAGE
    string imagename("0000.jpg");
    Mat im;
    im = imread(imagename.c_str(), IMREAD_COLOR);

    unsigned height = im.rows;
    unsigned width = im.cols;
#else 
    unsigned height = 32;
    unsigned width = 32;

#endif
    cout << "Image height: " << height << endl;
    cout << "Image width:  " << width << endl;

    const unsigned pixel_bits = L0_Ibit*L0_Cin;
    const unsigned pixels_per_line = 384/pixel_bits;
    const unsigned buffer_size = (NUM_SAMPLES*height*width)/pixels_per_line;
	stream<ap_axis > inputStream("inputStream");

    cout << "pixels_per_line: " << pixels_per_line << endl;
    cout << "buffer_size: " << buffer_size << endl;
	
#ifdef REAL_IMAGE
    uint8_t* pixel_ptr = (uint8_t*)im.data;
    unsigned channels = im.channels();
#else
    uint8_t* pixel_ptr = (uint8_t*)malloc(3*height*width);
    unsigned channels = 3;
    unsigned k = 0;
    for (unsigned y = 0; y < height; y++){
        for (unsigned x = 0; x < width; x++){
            for (unsigned c = 0; c < channels; c++){
                pixel_ptr[y*width*channels + x*channels + c] = (k++)%256;  
            } 
        }  
    }

#endif

	unsigned int index = 0;
    unsigned word;

    for (unsigned i = 0; i < NUM_SAMPLES; i++){
        word = 0;
        ap_axis temp;
        for (unsigned y = 0; y < height; y++){
            for (unsigned x = 0; x < width; x++){
                unsigned red = (unsigned)pixel_ptr[y*width*channels + x*channels];
                unsigned green = (unsigned)pixel_ptr[y*width*channels + x*channels + 1];
                unsigned blue = (unsigned)pixel_ptr[y*width*channels + x*channels + 2];
                unsigned rgb = (blue << 16) + (green << 8) + red;

                temp.data(pixel_bits*(word+1)-1, pixel_bits*word) = rgb;

                if (word == pixels_per_line-1){
                    inputStream.write(temp);
                    word = 0;
                    temp.data = 0;
                    index++;
                
                }
                else {
                    word++; 
                }
            
            }
        
        
        }  
    }

#ifndef REAL_IMAGE
    free(pixel_ptr);
#endif

    cout << "index: " << index << endl;
    cout << "word: " << word << endl;

	stream<ap_axis > outputStream;

	DoCompute(inputStream, outputStream, NUM_SAMPLES);


	ap_axis outputBuffer[NUM_SAMPLES];

	for (unsigned int i = 0; i < NUM_SAMPLES; i++) {
		outputBuffer[i] = outputStream.read();
	}
	
	unsigned long MASK = ((long)1 << L4_Mbit) - 1;
	unsigned int count_trues = 0;
	for (unsigned int i = 0; i < NUM_SAMPLES; i++) {
		cout << "outputBuffer[" << i << "]: " << hex << outputBuffer[i].data << dec << endl;
		int max = 0;
		unsigned int prediction = -1;
		for (unsigned int j = 0; j < L4_Dout; j++) {
			int temp = (outputBuffer[i].data >> (j*L4_Mbit)) & MASK;
			temp = temp >> SCALE_BITS;
			cout << "outputBuffer[" << i << "][" << j << "]: " << static_cast<int>(temp) << endl;
			if (temp > max){
				max = temp;
				prediction = j;
			}
		}
		cout << "prediction: " << prediction << endl;
	}
	// cout << count_trues << " correct out of " << NUM_SAMPLES << endl;
}
#endif
