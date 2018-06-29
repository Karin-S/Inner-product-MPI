Pairwise Inner Products of Row Vectors:  Consider a matrix of size N by M, and each row in the matrix is considered as a vector of length M. Pairwise inner products of row vectors are to calculate inner products of all possible pairs of different rows in the matrix. For N rows there are N*(N-1)/2 such pairs. Assume N is equal to 6, for example, we have the following 15 row pairs:
(0, 1) (0, 2) (0, 3) (0, 4) (0, 5)
(1, 2) (1, 3) (1, 4) (1, 5)
(2, 3) (2, 4) (2, 5)
(3, 4) (3, 5)
(4, 5)
You are asked to design a parallel algorithm and implement it using MPI non-blocking send/recv communication functions.
In the parallel algorithm design you must consider efficiency issues, that is, try to
minimize communication costs and
balance the workloads among all processes.
In the implementation
The processes are organized as a one-dimensional linear array.
For simplicity assume N, is divisible by p for p being the number of processes.
Again for simplicity you may restrict p to be either an odd, or even number to achieve the best possible load balancing.
You must use MPI non-blocking send/recv.
Your program must produce correct results for p being greater than or equal to one.
Your program needs to ask for the matrix sizes N and M as user defined parameters, and must print out the results in the same (row-wise) order as shown in the above example.
After the parallel computation, your main program must conduct a self-checking, i.e., first perform a sequential computation using the same data set and then compare the two results.
