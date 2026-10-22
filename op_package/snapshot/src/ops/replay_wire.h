#pragma once
#include <cstdint>
#include <cstddef>
namespace replay_wire {
constexpr uint32_t Magic=0x33525044;
constexpr unsigned T=16,H=8,D=128;
struct alignas(128) Header {
  uint32_t magic,version,rows,heads;
  float scales[5];
  int32_t zeros[5];
  int32_t valid[T];
  uint32_t reserved[2];
};
static_assert(sizeof(Header)==128,"wire header");
constexpr size_t StateBytes=H*D*D*2;
constexpr size_t VectorElements=T*H*D;
constexpr size_t GateElements=T*H;
constexpr size_t Bytes=sizeof(Header)+(2*VectorElements+2*GateElements)*2;
constexpr size_t Words=Bytes/4;
static_assert(Bytes%128==0,"aligned wire");
}
