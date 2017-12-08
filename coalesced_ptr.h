#pragma once

/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * * Neither the name of NVIDIA CORPORATION nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
  A smart pointer that automatically provide coalesced memory
  transcations for arrays of arbitrary structures.  Given a structure
  T, of size S bytes, e.g.,
  struct T {
    char a[S];
  }
  in an array with sites elements
  T t[sites];
  using a coalesced_ptr will split the structure for reading and
  writing to memory as an array of structures of array of structures (AoSoAoS),
  where:
    - the inner structure size is given by memory_word_size
    - the inner array size is given by site_vector
    - the outer structure size is given by sizeof(T)/memory_word_size
    - the outer array size is given by sites/site_vector
  If size_vector=0, then we do a full transposition with respect to
  the number of sites, and we have a structure of arrays of structures  (SoAoS)
    - the inner structure size is given by memory_word_size
    - the array size is given by sites
    - the outer structure size is given by sizeof(T)/memory_word_size
*/

#ifdef __CUDACC__
#define COALESCED_PTR_OFFLOAD __device__ __host__
#else
#define COALESCED_PTR_OFFLOAD
#endif

/* the granularity of memory transactions in bytes - current GPUs
   support 4 byte, 8 byte and 16 byte transactions per thread */
constexpr int memory_word_size = 4;

/* the granularity of (virtual) register storage size in bytes - all
   current GPUs use 4-byte registers so just leave at 4 */
constexpr int register_word_size = 4;


/* the size of the inner array size */
static constexpr int site_vector = 32;

struct alignas(8) _int2 { int x, y; };
struct alignas(16) _int4 { int x, y, z, w; };

template <typename A, typename B> COALESCED_PTR_OFFLOAD inline void copy(A *a, const B *b, int n) { printf("Bork\n"); }

template<> COALESCED_PTR_OFFLOAD inline void copy<int,int>(int *a, const int *b, int n)
{ for (int i=0; i<n; i++) a[i] = b[i]; }

template<> COALESCED_PTR_OFFLOAD inline void copy<_int2,_int2>(_int2 *a, const _int2 *b, int n)
{ for (int i=0; i<n; i++) a[i] = b[i]; }
template<> COALESCED_PTR_OFFLOAD inline void copy<_int2,int>(_int2 *a, const int *b, int n) {
  for (int i=0; i<n; i++) { a[i].x = b[2*i]; a[i].y = b[2*i+1]; }
}
template<> COALESCED_PTR_OFFLOAD inline void copy<_int4,int>(_int4 *a, const int *b, int n) {
  for (int i=0; i<n; i++) { a[i].x = b[4*i+0]; a[i].y = b[4*i+1]; a[i].z = b[4*i+2]; a[i].w = b[4*i+3]; }
}

template<> COALESCED_PTR_OFFLOAD inline void copy<_int4,_int2>(_int4 *a, const _int2 *b, int n) {
  for (int i=0; i<n; i++) { a[i].x = b[2*i+0].x; a[i].y = b[2*i+0].y; a[i].z = b[2*i+1].x; a[i].w = b[2*i+1].y; }
}

template<> COALESCED_PTR_OFFLOAD inline void copy<int,_int2>(int *a, const _int2 *b, int n) {
  for (int i=0; i<n/2; i++) { a[2*i+0] = b[i].x; a[2*i+1] = b[i].y; }
}

template<> COALESCED_PTR_OFFLOAD inline void copy<int,_int4>(int *a, const _int4 *b, int n) {
  for (int i=0; i<n/4; i++) { a[4*i+0] = b[i].x; a[4*i+1] = b[i].y; a[4*i+2] = b[i].z; a[4*i+3] = b[i].w; }
}

template<> COALESCED_PTR_OFFLOAD inline void copy<_int2,_int4>(_int2 *a, const _int4 *b, int n) {
  for (int i=0; i<n/2; i++) { a[2*i+0].x = b[i].x; a[2*i+0].y = b[i].y; a[2*i+1].x = b[i].z; a[2*i+1].y = b[i].w; }
}

template <int size> struct short_vector;
template <> struct short_vector<4> { typedef int type; };
template <> struct short_vector<8> { typedef _int2 type; };
template <> struct short_vector<16> { typedef _int4 type; };

typedef typename short_vector<memory_word_size>::type memory_word;
typedef typename short_vector<register_word_size>::type register_word;

#if 1

template<typename T>
struct coalesced_ref {
  T* m_ptr;
  const int sites;
  const int idx;

  static constexpr int memory_elements = sizeof(T)/memory_word_size;
  static constexpr int register_elements = sizeof(T)/register_word_size;
  static constexpr int N = register_elements / memory_elements;

  static_assert(memory_elements*memory_word_size == sizeof(T), "Data structure length must be divisible by memory word size");
  static_assert(register_elements*register_word_size == sizeof(T), "Data structure length must be divisible by register word size");
  static_assert(memory_word_size / register_word_size != 0, "Memory word size must be a multiplier of register word size");

  COALESCED_PTR_OFFLOAD inline explicit coalesced_ref(T* ptr, int sites, int idx) : m_ptr(ptr), sites(sites), idx(idx) {}

  COALESCED_PTR_OFFLOAD inline operator T() const {
    T t;
    memory_word *mem_ptr = reinterpret_cast<memory_word*>(m_ptr);
    memory_word t_mem[ memory_elements ];
    if (site_vector) {
      for (int i=0; i<memory_elements; i++) t_mem[i] = mem_ptr[ ((idx/site_vector)*memory_elements + i)*site_vector + idx%site_vector ];
    } else {
      for (int i=0; i<memory_elements; i++) t_mem[i] = mem_ptr[ i*sites + idx ];
    }
    copy(reinterpret_cast<register_word*>(&t), t_mem, register_elements);
    return t;
  }

  COALESCED_PTR_OFFLOAD inline coalesced_ref& operator=(const T& t) {
    memory_word *mem_ptr = reinterpret_cast<memory_word*>(m_ptr);
    const memory_word *t_ptr = reinterpret_cast<const memory_word*>(&t);
    if (site_vector) {
      for (int i=0; i<memory_elements; i++) mem_ptr[ ((idx/site_vector)*memory_elements + i)*site_vector + idx%site_vector ] = t_ptr[i];
    } else {
      for (int i=0; i<memory_elements; i++) mem_ptr[ i*sites + idx ] = t_ptr[i];
    }
    return *this;
  }

  COALESCED_PTR_OFFLOAD inline const coalesced_ref& operator=(const T& t) const {
    memory_word *mem_ptr = reinterpret_cast<memory_word*>(m_ptr);
    const memory_word *t_ptr = reinterpret_cast<const memory_word*>(&t);
    if (site_vector) {
      for (int i=0; i<memory_elements; i++) mem_ptr[ ((idx/site_vector)*memory_elements + i)*site_vector + idx%site_vector ] = t_ptr[i];
    } else {
      for (int i=0; i<memory_elements; i++) mem_ptr[ i*sites + idx ] = t_ptr[i];
    }
    return *this;
  }

  COALESCED_PTR_OFFLOAD inline coalesced_ref& operator=(const coalesced_ref& other) {
    const memory_word *other_ptr = reinterpret_cast<const memory_word*>(other.m_ptr);
    memory_word *this_ptr = reinterpret_cast<memory_word*>(m_ptr);
    if (site_vector) {
      for (int i=0; i<memory_elements; i++) this_ptr[ ((idx/site_vector)*memory_elements + i)*site_vector + idx%site_vector ] =
					      other_ptr[ ((idx/site_vector)*memory_elements + i)*site_vector + idx%site_vector ];
    } else {
      for (int i=0; i<memory_elements; i++) this_ptr[ i*sites + idx ] = other_ptr[ i*sites + idx ];
    }
    return *this;
  }
};

#else

template<typename T>
struct coalesced_ref {
  T* m_ptr;
  const int sites;
  const int idx;
  COALESCED_PTR_OFFLOAD inline explicit coalesced_ref(T* ptr, int sites, int idx) : m_ptr(ptr), sites(sites), idx(idx) {}
  COALESCED_PTR_OFFLOAD inline operator T() const { return *(m_ptr+idx); }
  COALESCED_PTR_OFFLOAD inline coalesced_ref& operator=(const T& t) { *(m_ptr+idx) = t; return *this; }
  COALESCED_PTR_OFFLOAD inline const coalesced_ref& operator=(const T& t) const { *(m_ptr+idx) = t; return *this; }
  COALESCED_PTR_OFFLOAD inline coalesced_ref& operator=(const coalesced_ref& other) { *(other.m_ptr+idx) = *(m_ptr+idx); return *this; }
};

#endif

template<typename T>
struct coalesced_ptr {
  T* m_ptr;
  int sites;

  COALESCED_PTR_OFFLOAD inline coalesced_ptr(T* ptr, int sites_) : m_ptr(ptr), sites(sites_) {}
  COALESCED_PTR_OFFLOAD inline coalesced_ptr() : m_ptr(nullptr), sites(0) {}

  template<typename I>
  COALESCED_PTR_OFFLOAD inline coalesced_ref<T> operator[](const I& idx) {
    return coalesced_ref<T>(m_ptr,sites,idx);
  }

  template<typename I>
  COALESCED_PTR_OFFLOAD inline const coalesced_ref<T> operator[](const I& idx) const {
    return coalesced_ref<T>(m_ptr,sites,idx);
  }
};


#ifdef __CUDACC__
#undef COALESCED_PTR_OFFLOAD
#endif
