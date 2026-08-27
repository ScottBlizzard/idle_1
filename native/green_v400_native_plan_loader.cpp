#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unordered_map>
#include <unistd.h>
#include <vector>

namespace {

constexpr std::size_t kHeaderBytes = 256;
constexpr std::size_t kMaxDescriptorBytes = kHeaderBytes + 64U * 1024U * 1024U;
constexpr std::uint8_t kMagic[16] = {
    'G','R','E','E','N','V','4','0','0','D','E','S','C',0,0,0};

std::uint32_t rotr(std::uint32_t x, std::uint32_t n) { return (x >> n) | (x << (32U - n)); }

class Sha256 {
 public:
  Sha256() : state_{0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,
                    0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U} {}
  void update(const std::uint8_t* data, std::size_t size) {
    total_ += size;
    while (size) {
      const std::size_t take = std::min(size, block_.size() - used_);
      std::memcpy(block_.data() + used_, data, take);
      used_ += take; data += take; size -= take;
      if (used_ == block_.size()) { transform(block_.data()); used_ = 0; }
    }
  }
  std::array<std::uint8_t,32> finish() {
    const std::uint64_t bits = static_cast<std::uint64_t>(total_) * 8U;
    block_[used_++] = 0x80U;
    if (used_ > 56U) {
      while (used_ < 64U) block_[used_++] = 0;
      transform(block_.data()); used_ = 0;
    }
    while (used_ < 56U) block_[used_++] = 0;
    for (int shift = 56; shift >= 0; shift -= 8) block_[used_++] = (bits >> shift) & 0xffU;
    transform(block_.data());
    std::array<std::uint8_t,32> result{};
    for (std::size_t i = 0; i < 8; ++i)
      for (std::size_t j = 0; j < 4; ++j)
        result[i*4+j] = (state_[i] >> (24U - 8U*j)) & 0xffU;
    return result;
  }
 private:
  void transform(const std::uint8_t* input) {
    static constexpr std::uint32_t k[64] = {
      0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
      0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
      0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
      0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
      0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
      0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
      0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
      0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U};
    std::uint32_t w[64];
    for (std::size_t i=0;i<16;++i) w[i]=(std::uint32_t(input[i*4])<<24)|(std::uint32_t(input[i*4+1])<<16)|(std::uint32_t(input[i*4+2])<<8)|input[i*4+3];
    for (std::size_t i=16;i<64;++i) {
      const std::uint32_t s0=rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3);
      const std::uint32_t s1=rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10);
      w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    std::uint32_t a=state_[0],b=state_[1],c=state_[2],d=state_[3],e=state_[4],f=state_[5],g=state_[6],h=state_[7];
    for (std::size_t i=0;i<64;++i) {
      const std::uint32_t s1=rotr(e,6)^rotr(e,11)^rotr(e,25), ch=(e&f)^((~e)&g);
      const std::uint32_t t1=h+s1+ch+k[i]+w[i], s0=rotr(a,2)^rotr(a,13)^rotr(a,22), maj=(a&b)^(a&c)^(b&c);
      const std::uint32_t t2=s0+maj; h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    state_[0]+=a;state_[1]+=b;state_[2]+=c;state_[3]+=d;state_[4]+=e;state_[5]+=f;state_[6]+=g;state_[7]+=h;
  }
  std::array<std::uint32_t,8> state_; std::array<std::uint8_t,64> block_{};
  std::size_t used_=0,total_=0;
};

bool parse_hex(const char* text, std::array<std::uint8_t,32>& output) {
  if (!text || std::strlen(text)!=64) return false;
  for (std::size_t i=0;i<32;++i) {
    auto nib=[](char c)->int { if(c>='0'&&c<='9')return c-'0'; if(c>='a'&&c<='f')return c-'a'+10; return -1; };
    int hi=nib(text[i*2]),lo=nib(text[i*2+1]); if(hi<0||lo<0)return false; output[i]=(hi<<4)|lo;
  } return true;
}
std::uint32_t u32(const std::uint8_t* p){return std::uint32_t(p[0])|(std::uint32_t(p[1])<<8)|(std::uint32_t(p[2])<<16)|(std::uint32_t(p[3])<<24);}
std::uint64_t u64(const std::uint8_t* p){return std::uint64_t(u32(p))|(std::uint64_t(u32(p+4))<<32);}
bool read_all(int fd,std::vector<std::uint8_t>& out){std::size_t n=0;while(n<out.size()){ssize_t r=::pread(fd,out.data()+n,out.size()-n,n);if(r<=0)return false;n+=r;}return true;}

struct PlanEnvelope { int dfd=-1,bfd=-1; void* blob=MAP_FAILED; std::uint64_t dsize=0,bsize=0; std::uint32_t records=0,nodes=0,bindings=0,fusion=0; ~PlanEnvelope(){if(blob!=MAP_FAILED)::munmap(blob,bsize);if(dfd>=0)::close(dfd);if(bfd>=0)::close(bfd);} };
std::mutex registry_mutex; std::unordered_map<std::uint64_t,std::unique_ptr<PlanEnvelope>> registry; std::atomic<std::uint64_t> next_handle{1};

}  // namespace

extern "C" int green_v400_native_plan_envelope_open_v1(
    const char* descriptor_path,const char* blob_path,const char* expected_descriptor_sha,
    const char* expected_program_sha,const char* expected_dispatch_sha,const char* expected_blob_sha,
    const char* expected_fusion_sha,std::uint64_t expected_blob_nbytes,
    std::uint32_t expected_fusion_weights,std::uint64_t* out_handle) {
  if(!descriptor_path||!blob_path||!out_handle)return 2;*out_handle=0;
  std::array<std::uint8_t,32> ed,ep,et,eb,ef;
  if(!parse_hex(expected_descriptor_sha,ed)||!parse_hex(expected_program_sha,ep)||!parse_hex(expected_dispatch_sha,et)||!parse_hex(expected_blob_sha,eb)||!parse_hex(expected_fusion_sha,ef))return 2;
  std::unique_ptr<PlanEnvelope> plan(new PlanEnvelope());
  plan->dfd=::open(descriptor_path,O_RDONLY|O_CLOEXEC|O_NOFOLLOW); if(plan->dfd<0)return 3;
  struct stat ds{}; if(::fstat(plan->dfd,&ds)!=0||!S_ISREG(ds.st_mode)||ds.st_size<(off_t)kHeaderBytes||ds.st_size>(off_t)kMaxDescriptorBytes)return 4;
  plan->dsize=ds.st_size; std::vector<std::uint8_t> bytes(plan->dsize); if(!read_all(plan->dfd,bytes))return 4;
  Sha256 whole;whole.update(bytes.data(),bytes.size());if(whole.finish()!=ed)return 5;
  const std::uint8_t* h=bytes.data();
  if(std::memcmp(h,kMagic,16)!=0||u32(h+16)!=1||u32(h+20)!=256||u32(h+24)!=0x01020304U||u32(h+28)!=64||u64(h+32)!=256||u64(h+40)!=bytes.size()-256)return 6;
  if(std::memcmp(h+80,ep.data(),32)||std::memcmp(h+112,et.data(),32)||std::memcmp(h+144,eb.data(),32)||std::memcmp(h+176,ef.data(),32))return 6;
  plan->records=u32(h+208);plan->nodes=u32(h+212);plan->bindings=u32(h+216);plan->fusion=u32(h+220);
  if(plan->records!=32||plan->nodes!=81||plan->bindings!=150
      ||expected_fusion_weights==0||expected_fusion_weights>16384
      ||plan->fusion!=expected_fusion_weights)return 6;
  for(std::size_t i=224;i<256;++i)if(h[i]!=0)return 6;
  Sha256 payload;payload.update(bytes.data()+256,bytes.size()-256);auto pd=payload.finish();if(std::memcmp(h+48,pd.data(),32))return 7;
  plan->bfd=::open(blob_path,O_RDONLY|O_CLOEXEC|O_NOFOLLOW);if(plan->bfd<0)return 3;struct stat bs{};
  if(::fstat(plan->bfd,&bs)!=0||!S_ISREG(bs.st_mode)||std::uint64_t(bs.st_size)!=expected_blob_nbytes)return 8;plan->bsize=bs.st_size;
  Sha256 blob_hash;std::vector<std::uint8_t> chunk(1U<<20);std::uint64_t off=0;while(off<plan->bsize){std::size_t want=std::min<std::uint64_t>(chunk.size(),plan->bsize-off);ssize_t n=::pread(plan->bfd,chunk.data(),want,off);if(n<=0)return 8;blob_hash.update(chunk.data(),n);off+=n;}if(blob_hash.finish()!=eb)return 8;
  plan->blob=::mmap(nullptr,plan->bsize,PROT_READ,MAP_PRIVATE,plan->bfd,0);if(plan->blob==MAP_FAILED)return 9;
  std::uint64_t handle=next_handle.fetch_add(1);if(handle==0)return 10;{std::lock_guard<std::mutex> lock(registry_mutex);registry.emplace(handle,std::move(plan));}*out_handle=handle;return 0;
}

extern "C" int green_v400_native_plan_envelope_info_v1(std::uint64_t handle,std::uint64_t* descriptor_nbytes,std::uint64_t* blob_nbytes,std::uint32_t* records,std::uint32_t* nodes,std::uint32_t* bindings,std::uint32_t* fusion_weights){std::lock_guard<std::mutex> lock(registry_mutex);auto it=registry.find(handle);if(it==registry.end())return 2;auto& p=*it->second;if(descriptor_nbytes)*descriptor_nbytes=p.dsize;if(blob_nbytes)*blob_nbytes=p.bsize;if(records)*records=p.records;if(nodes)*nodes=p.nodes;if(bindings)*bindings=p.bindings;if(fusion_weights)*fusion_weights=p.fusion;return 0;}
extern "C" int green_v400_native_plan_envelope_close_v1(std::uint64_t handle){std::unique_ptr<PlanEnvelope> removed;{std::lock_guard<std::mutex> lock(registry_mutex);auto it=registry.find(handle);if(it==registry.end())return 2;removed=std::move(it->second);registry.erase(it);}return 0;}
