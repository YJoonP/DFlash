#include "../src/ops/replay_wire.h"
uint32_t execute_replay(QHPI_RuntimeHandle*, uint32_t no,QHPI_Tensor** outputs,
                        uint32_t ni,const QHPI_Tensor* const* inputs) {
#ifndef DR16M_HAVE_HVX
  return 1;
#else
  using namespace replay_wire;
  if(ni!=3 || no!=1 || !shape(inputs[0],1,1,1,Words) ||
     !shape(inputs[1],1,H,D,D/2) || !shape(inputs[2],1,1,1,4) || !shape(outputs[0],1,H,D,D/2)) return 1;
  auto* raw=static_cast<const uint8_t*>(qhpi_tensor_raw_data(inputs[0]));
  auto* ctrl=static_cast<const int32_t*>(qhpi_tensor_raw_data(inputs[2]));
  auto* dst=static_cast<uint16_t*>(qhpi_tensor_raw_data(outputs[0]));
  if(!raw || !ctrl || !dst || ctrl[0]<0 || ctrl[0]>int(T)) return 1;
  Header header;std::memcpy(&header,raw,sizeof(header));
  if(header.magic!=Magic || header.version!=2 || header.rows!=T || header.heads!=H) return 1;
  for(int i=0;i<5;++i) if(!(header.scales[i]>0) || !std::isfinite(header.scales[i]) || header.zeros[i]<0 || header.zeros[i]>65535) return 1;
  auto* base=static_cast<const uint16_t*>(qhpi_tensor_raw_data(inputs[1]));
  if(!base)return 1;
  auto* keys=reinterpret_cast<const uint16_t*>(raw+sizeof(Header));
  auto* values=keys+VectorElements;
  auto* betas=values+VectorElements;
  auto* decays=betas+GateElements;
  alignas(128) float k[128],v[128];
  alignas(128) dr16m::Tables tb;
  for(unsigned h=0;h<H;++h) {
    const uint16_t* src=base+h*D*D;
    uint16_t* out=dst+h*D*D;
    bool wrote=false;
    for(int t=0;t<ctrl[0];++t) {
      if(!header.valid[t]) continue;
      const size_t row=t*H+h;
      deltarule::load_u16(keys+row*D,k,D,header.zeros[0],header.scales[0]);
      deltarule::load_u16(values+row*D,v,D,header.zeros[1],header.scales[1]);
      const float beta=(int(betas[row])-header.zeros[2])*header.scales[2];
      const float decay=(int(decays[row])-header.zeros[3])*header.scales[3];
      dr16m::step_state_hvx(src,out,header.zeros[4],header.scales[4],1.f/header.scales[4],
                           dr16m::lm_of(header.zeros[4]),k,v,decay,beta,h,tb);
      src=out;wrote=true;
    }
    if(!wrote) snapshot_copy(out,src,D*D*2);
  }
  return 0;
#endif
}
