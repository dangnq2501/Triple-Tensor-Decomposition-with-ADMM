********************************************************************************************
    Fully-Connected Tensor Network Decomposition and Its Application 
                      to Higher-Order Tensor Completion
********************************************************************************************

         Copyright:  Yu-Bang Zheng, Ting-Zhu Huang,  Xi-Le Zhao, 
                                   Qibin Zhao, and Tai-Xiang Jiang

**Please cite our paper [1] if you use any part of our code**

 1). Get Started

 Run the 'Demo_HSV.m' to run the FCTN decomposition-based TC method.

 
 2). Details

 An illustration of two main codes: 'inc_FCTN_TC.m' and 'inc_FCTN_TC_end.m'

 'inc_FCTN_TC.m' supposes that the size of the factor G_k is 
  R_{1,k}*R_{2,k}*...*R_{k-1,k}*I_k*R_{k,k+1}*...*R_{k,N}. (consistent with the paper)

 'inc_FCTN_TC_end.m' supposes that the size of the factor G_k is 
  R_{1,k}*R_{2,k}*...*R_{k-1,k}*R_{k,k+1}*...*R_{k,N}*I_k. (can set the FCTN rank as 1 in Matlab)

 More details can be found in [1]

    [1]  Yu-Bang Zheng, Ting-Zhu Huang, Xi-Le Zhao, Qibin Zhao, Tai-Xiang Jiang, 
          Fully-Connected Tensor Network Decomposition and Its Application to Higher-Order 
          Tensor Completion, AAAI 2021, vol. 35, no. 12, pp. 11071-11078, 2021.

 3). Citations

    The data is available at: http://openremotesensing.net/kb/data/. [2]

    [2] Mian, A.; and Hartley, R., Hyperspectral video restoration using optical flow and sparse coding.
         Optics Express 20(10):10658¨C10673, 2012.

