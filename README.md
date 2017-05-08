# coalesced_ptr

A smart pointer that automatically provides coalesced memory
transcations for arrays of arbitrary structures.  Given a structure
T, of size S bytes, e.g.,
```c++
struct T {                                                                                                                     
  char a[S];                                                                                                                   
};
```
in an array with `sites` elements
```c++
T t[sites];
```
using a coalesced_ptr will split the structure for reading writing to
memory as an array of structures of array of structures (AoSoAoS),
where:

* the inner structure size is given by memory_word_size
* the inner array size is given by site_vector
* the outer structure size is given by sizeof(T)/memory_word_size
* the outer array size is given by sites/site_vector

If `size_vector=0`, then we do a full transposition with respect to
the number of sites, and we have a structure of arrays of structures
(SoAoS)

* the inner structure size is given by memory_word_size                                                                      
* the array size is given by sites                                                                                           
* the outer structure size is given by sizeof(T)/memory_word_size

Note that there is the implicit assumption that the structure size is uniform along all sites.