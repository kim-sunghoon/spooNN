// conv0/Conv2D
// Cycles per IFM: 36864.0
#define L0_K 3
#define L0_S 1
#define L0_Din 32
#define L0_Cin 3
#define L0_Cout 32
#define L0_Ibit 8
#define L0_Wbit 20
#define L0_Mbit 32
#define L0_Abit 5
#define L0_SWU_OutP 1
#define L0_MVTU_InP 3
#define L0_MVTU_OutP 8

// conv1/Conv2D
// Cycles per IFM: 36864.0
#define L1_K 3
#define L1_S 1
#define L1_Din 16
#define L1_Cin 32
#define L1_Cout 32
#define L1_Ibit 5
#define L1_Wbit 1
#define L1_Mbit 32
#define L1_Abit 5
#define L1_SWU_OutP 1
#define L1_MVTU_InP 8
#define L1_MVTU_OutP 8

// conv2/Conv2D
// Cycles per IFM: 36864.0
#define L2_K 3
#define L2_S 1
#define L2_Din 16
#define L2_Cin 32
#define L2_Cout 32
#define L2_Ibit 5
#define L2_Wbit 1
#define L2_Mbit 32
#define L2_Abit 5
#define L2_SWU_OutP 1
#define L2_MVTU_InP 8
#define L2_MVTU_OutP 8

// fc0/MatMul
// Cycles per IFM: 4096.0
#define L3_Din 2048
#define L3_Dout 512
#define L3_Ibit 5
#define L3_Wbit 1
#define L3_Mbit 32
#define L3_Abit 5
#define L3_InP 32
#define L3_OutP 8

// fc1/MatMul
// Cycles per IFM: 64.0
#define L4_Din 512
#define L4_Dout 10
#define L4_Ibit 5
#define L4_Wbit 20
#define L4_Mbit 32
#define L4_Abit 5
#define L4_InP 8
#define L4_OutP 10

// pool0/max_pooling2d/MaxPool
// Cycles per IFM: 1632.0
#define L5_K 2
#define L5_S 2
#define L5_Din 32
#define L5_Cin 32
#define L5_Ibit 5
#define L5_SWU_OutP 1

// pool1/max_pooling2d/MaxPool
// Cycles per IFM: 432.0
#define L6_K 2
#define L6_S 2
#define L6_Din 16
#define L6_Cin 32
#define L6_Ibit 5
#define L6_SWU_OutP 1

#define SCALE_BITS 18
#define FACTOR_SCALE_BITS 22
#define HIGH_PREC_SCALE_BITS 22

// #pragma HLS RESOURCE variable=weights0 core=RAM_1P_BRAM
// #pragma HLS ARRAY_PARTITION variable=weights0 complete dim=0
// #pragma HLS RESOURCE variable=factorA0 core=RAM_1P_BRAM
// #pragma HLS RESOURCE variable=factorB0 core=RAM_1P_BRAM
// #pragma HLS ARRAY_PARTITION variable=factorA0 complete dim=0
// #pragma HLS ARRAY_PARTITION variable=factorB0 complete dim=0
// #pragma HLS RESOURCE variable=weights1 core=RAM_1P_BRAM
// #pragma HLS ARRAY_PARTITION variable=weights1 complete dim=0
// #pragma HLS RESOURCE variable=factorA1 core=RAM_1P_BRAM
// #pragma HLS RESOURCE variable=factorB1 core=RAM_1P_BRAM
// #pragma HLS ARRAY_PARTITION variable=factorA1 complete dim=0
// #pragma HLS ARRAY_PARTITION variable=factorB1 complete dim=0
// #pragma HLS RESOURCE variable=weights2 core=RAM_1P_BRAM
// #pragma HLS ARRAY_PARTITION variable=weights2 complete dim=0
// #pragma HLS RESOURCE variable=factorA2 core=RAM_1P_BRAM
// #pragma HLS RESOURCE variable=factorB2 core=RAM_1P_BRAM
// #pragma HLS ARRAY_PARTITION variable=factorA2 complete dim=0
// #pragma HLS ARRAY_PARTITION variable=factorB2 complete dim=0
// #pragma HLS RESOURCE variable=weights3 core=RAM_1P_BRAM
// #pragma HLS ARRAY_PARTITION variable=weights3 complete dim=0
// #pragma HLS RESOURCE variable=factorA3 core=RAM_1P_BRAM
// #pragma HLS RESOURCE variable=factorB3 core=RAM_1P_BRAM
// #pragma HLS ARRAY_PARTITION variable=factorA3 complete dim=0
// #pragma HLS ARRAY_PARTITION variable=factorB3 complete dim=0
// #pragma HLS RESOURCE variable=weights4 core=RAM_1P_BRAM
// #pragma HLS ARRAY_PARTITION variable=weights4 complete dim=0
// #pragma HLS RESOURCE variable=factorA4 core=RAM_1P_BRAM
// #pragma HLS RESOURCE variable=factorB4 core=RAM_1P_BRAM
// #pragma HLS ARRAY_PARTITION variable=factorA4 complete dim=0
// #pragma HLS ARRAY_PARTITION variable=factorB4 complete dim=0

