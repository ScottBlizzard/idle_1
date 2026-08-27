#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <memory>
#include <limits>
#include <string>
#include <unordered_set>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unordered_map>
#include <unistd.h>
#include <vector>

extern "C" int green_v400_resident_jet_buffer_import_f32_constants(
    std::uint32_t precision_bits,std::uint32_t width,
    const std::uint32_t* value_bits,void** output_handle);
extern "C" void green_v400_resident_jet_buffer_free(void* input_handle);

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
bool valid_utf8(const std::uint8_t* data,std::size_t size){
  std::size_t i=0;
  while(i<size){
    const std::uint8_t first=data[i++];if(first<=0x7fU)continue;
    std::uint32_t value=0;std::size_t continuation=0;
    if(first>=0xc2U&&first<=0xdfU){value=first&0x1fU;continuation=1;}
    else if(first>=0xe0U&&first<=0xefU){value=first&0x0fU;continuation=2;}
    else if(first>=0xf0U&&first<=0xf4U){value=first&0x07U;continuation=3;}
    else return false;
    if(continuation>size-i)return false;
    for(std::size_t j=0;j<continuation;++j){const std::uint8_t next=data[i++];
      if((next&0xc0U)!=0x80U)return false;value=(value<<6U)|(next&0x3fU);}
    if((continuation==2&&(value<0x800U||(value>=0xd800U&&value<=0xdfffU)))
        ||(continuation==3&&(value<0x10000U||value>0x10ffffU)))return false;
  }
  return true;
}

struct BinaryValue {
  enum Kind { kNull, kBool, kInt, kString, kList, kDict } kind = kNull;
  bool boolean = false; std::int64_t integer = 0; std::string string;
  std::vector<BinaryValue> list;
  std::vector<std::pair<std::string, BinaryValue>> dict;
  const std::uint8_t* encoded = nullptr;
  std::size_t encoded_size = 0;
};

class BinaryDecoder {
 public:
  BinaryDecoder(const std::uint8_t* data,std::size_t size):data_(data),size_(size){}
  bool parse(BinaryValue& output){return value(output,0)&&offset_==size_;}
 private:
  bool take(std::size_t count,const std::uint8_t*& out){if(count>size_-offset_)return false;out=data_+offset_;offset_+=count;return true;}
  bool count(std::uint32_t& out){const std::uint8_t* p;if(!take(4,p))return false;out=u32(p);return true;}
  bool value(BinaryValue& out,std::uint32_t depth){
    const std::size_t start=offset_;
    if(!value_body(out,depth))return false;
    out.encoded=data_+start;out.encoded_size=offset_-start;return true;
  }
  bool value_body(BinaryValue& out,std::uint32_t depth){
    if(depth>128||++items_>200000)return false;const std::uint8_t* p;if(!take(1,p))return false;
    const char tag=static_cast<char>(*p);
    if(tag=='N'){out.kind=BinaryValue::kNull;return true;}
    if(tag=='F'||tag=='T'){out.kind=BinaryValue::kBool;out.boolean=tag=='T';return true;}
    if(tag=='I'){
      const std::uint8_t* sign;if(!take(1,sign))return false;std::uint32_t n;if(!count(n)||n>8||*sign>1)return false;
      const std::uint8_t* raw;if(!take(n,raw)||(n&&raw[0]==0)||(!n&&*sign))return false;std::uint64_t magnitude=0;
      for(std::uint32_t i=0;i<n;++i)magnitude=(magnitude<<8)|raw[i];
      if(magnitude>static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))return false;
      out.kind=BinaryValue::kInt;out.integer=*sign?-static_cast<std::int64_t>(magnitude):static_cast<std::int64_t>(magnitude);return true;
    }
    if(tag=='S'){
      std::uint32_t n;if(!count(n)||n>(1U<<20))return false;const std::uint8_t* raw;if(!take(n,raw))return false;
      if(!valid_utf8(raw,n))return false;
      out.kind=BinaryValue::kString;out.string.assign(reinterpret_cast<const char*>(raw),n);return true;
    }
    if(tag=='L'){
      std::uint32_t n;if(!count(n)||n>20000)return false;out.kind=BinaryValue::kList;out.list.resize(n);
      for(auto& item:out.list)if(!value(item,depth+1))return false;return true;
    }
    if(tag=='D'){
      std::uint32_t n;if(!count(n)||n>20000)return false;out.kind=BinaryValue::kDict;out.dict.reserve(n);std::string prior;
      for(std::uint32_t i=0;i<n;++i){BinaryValue key,item;if(!value(key,depth+1)||key.kind!=BinaryValue::kString||(!prior.empty()&&key.string<=prior)||!value(item,depth+1))return false;prior=key.string;out.dict.emplace_back(std::move(key.string),std::move(item));}return true;
    }
    return false;
  }
  const std::uint8_t* data_;std::size_t size_,offset_=0,items_=0;
};

const BinaryValue* field(const BinaryValue& value,const char* name){if(value.kind!=BinaryValue::kDict)return nullptr;for(const auto& item:value.dict)if(item.first==name)return &item.second;return nullptr;}
bool integer(const BinaryValue* value,std::int64_t& out){if(!value||value->kind!=BinaryValue::kInt)return false;out=value->integer;return true;}
bool text(const BinaryValue* value,std::string& out){if(!value||value->kind!=BinaryValue::kString)return false;out=value->string;return true;}
bool sha(const BinaryValue* value){std::string s;if(!text(value,s)||s.size()!=64)return false;for(char c:s)if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))return false;return true;}
bool text_matches_digest(const BinaryValue* value,const std::uint8_t* digest){
  std::string expected;if(!sha(value)||!text(value,expected))return false;std::array<std::uint8_t,32> bytes{};
  if(!parse_hex(expected.c_str(),bytes))return false;return std::memcmp(bytes.data(),digest,32)==0;
}
std::size_t dtype_size(const std::string& dtype){if(dtype=="|u1"||dtype=="|i1")return 1;if(dtype=="<i2"||dtype=="<f2")return 2;if(dtype=="<i4"||dtype=="<f4")return 4;if(dtype=="<i8"||dtype=="<f8")return 8;return 0;}

struct NativeRecord {
  std::string name,dtype,tensor_semantic_sha256,data_sha256;
  std::uint64_t offset=0,nbytes=0;
  std::vector<std::uint32_t> shape;
};
struct NativeNode {
  std::string semantic_id,kernel_id;
  std::uint32_t kernel_tag=0;
  bool depends_on_t=false;
  std::int32_t axis=-1;
  std::uint32_t operation_tag=0,float_kernel_tag=0,final_position=0;
  std::uint32_t n_heads=0,d_head=0,contrast_width=0;
  std::vector<std::uint32_t> parents;
  std::vector<std::string> tensor_hashes;
  std::string output_dtype;
  std::vector<std::uint32_t> selected_gates,output_shape,required_axis0_rows;
};
struct NativeBinding {
  std::uint32_t node_index=0,input_ordinal=0,source_index=0;
  bool source_is_record=false;
  std::string tensor_semantic_sha256,source_name;
};
struct NativeDyadic { std::string significand; std::int64_t exponent_2=0; };
struct NativePlanTables {
  std::uint32_t sequence_length=0,d_model=0,d_mlp=0,n_heads=0,d_head=0;
  std::uint32_t final_position=0,contrast_width=0;
  std::vector<std::uint32_t> selected_gates;
  std::vector<NativeRecord> records;
  std::vector<NativeNode> nodes;
  std::vector<NativeBinding> bindings;
  std::vector<NativeDyadic> fusion_weights;
  NativeDyadic fusion_bias;
  std::array<std::uint32_t,4> branch_roots{};
  std::uint32_t output_root=0;
  std::size_t total_liveness_rows=0;
};
std::uint32_t kernel_tag(const std::string& kernel){
  if(kernel=="affine_scatter.v1")return 1;if(kernel=="static_view.v1")return 2;
  if(kernel=="pairwise_affine.v1")return 3;if(kernel=="layer_norm.v1")return 4;
  if(kernel=="gelu_new.v1")return 5;if(kernel=="causal_attention.v1")return 6;
  if(kernel=="residual_add.v1")return 7;if(kernel=="final_contrast.v1")return 8;
  if(kernel=="branch_linear_combination.v1")return 9;return 0;
}
bool dyadic(const BinaryValue* value,NativeDyadic& out){
  std::string significand;std::int64_t exponent;
  if(!value||value->kind!=BinaryValue::kDict||value->dict.size()!=2
      ||!text(field(*value,"significand"),significand)||significand.empty()
      ||!integer(field(*value,"exponent_2"),exponent))return false;
  std::size_t start=significand[0]=='-'?1:0;if(start==significand.size())return false;
  if(significand[start]=='0'&&significand.size()-start!=1)return false;
  for(std::size_t i=start;i<significand.size();++i)if(significand[i]<'0'||significand[i]>'9')return false;
  if(significand=="0"&&exponent!=0)return false;
  if(significand!="0"&&((significand.back()-'0')%2)==0)return false;
  out.significand=std::move(significand);out.exponent_2=exponent;return true;
}
bool boolean(const BinaryValue* value,bool& out){if(!value||value->kind!=BinaryValue::kBool)return false;out=value->boolean;return true;}
bool string_is(const BinaryValue* value,const char* expected){std::string actual;return text(value,actual)&&actual==expected;}
bool encoded_equal(const BinaryValue* first,const BinaryValue* second){return first&&second&&first->encoded_size==second->encoded_size
    &&std::memcmp(first->encoded,second->encoded,first->encoded_size)==0;}
bool validate_node_attrs(const BinaryValue* attrs,const BinaryValue* output_spec,
                         std::uint32_t final_position,std::uint32_t n_heads,
                         std::uint32_t d_head,std::uint32_t contrast_width,
                         const std::vector<std::uint32_t>& selected_gates,NativeNode& node){
  if(!attrs||attrs->kind!=BinaryValue::kDict||!boolean(field(*attrs,"depends_on_t"),node.depends_on_t))return false;
  const BinaryValue* mask=field(*attrs,"dependency_mask_spec");
  if(!mask||mask->kind!=BinaryValue::kDict||mask->dict.size()!=5
      ||!string_is(field(*mask,"schema_version"),"green-v400-dependency-mask-v2")
      ||!encoded_equal(field(*mask,"output_spec"),output_spec))return false;
  std::string kind;std::int64_t dependent_count;
  const BinaryValue* indices=field(*mask,"axis0_indices");
  if(!text(field(*mask,"kind"),kind)||!integer(field(*mask,"dependent_scalar_count"),dependent_count)
      ||dependent_count<0||!indices||indices->kind!=BinaryValue::kList)return false;
  std::uint64_t trailing=1;for(std::size_t i=1;i<node.output_shape.size();++i){if(node.output_shape[i]
      &&trailing>std::numeric_limits<std::uint64_t>::max()/node.output_shape[i])return false;trailing*=node.output_shape[i];}
  std::int64_t prior=-1;for(const auto& index:indices->list){if(index.kind!=BinaryValue::kInt||index.integer<=prior
      ||node.output_shape.empty()||index.integer<0||index.integer>=node.output_shape[0])return false;prior=index.integer;}
  std::uint64_t scalar_count=1;for(std::uint32_t dim:node.output_shape){if(dim
      &&scalar_count>std::numeric_limits<std::uint64_t>::max()/dim)return false;scalar_count*=dim;}
  if(scalar_count>static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
      ||trailing>static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
      ||indices->list.size()>static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())/std::max<std::uint64_t>(trailing,1))return false;
  if((kind=="empty"&&(!indices->list.empty()||dependent_count!=0||node.depends_on_t))
      ||(kind=="dense"&&(!indices->list.empty()||dependent_count!=static_cast<std::int64_t>(scalar_count)||!node.depends_on_t))
      ||(kind=="axis0_rows"&&(indices->list.empty()||dependent_count!=static_cast<std::int64_t>(indices->list.size()*trailing)||!node.depends_on_t))
      ||(kind!="empty"&&kind!="dense"&&kind!="axis0_rows"))return false;
  std::int64_t value;
  if(node.kernel_id=="affine_scatter.v1"){
    if(attrs->dict.size()!=4||!string_is(field(*attrs,"control"),"same_t_times_physical_direction")
        ||!integer(field(*attrs,"final_position"),value)||value!=final_position)return false;
    node.final_position=final_position;return true;
  }
  if(node.kernel_id=="static_view.v1"){
    std::string operation;if(!text(field(*attrs,"operation"),operation))return false;
    if(operation=="tensor_constant")node.operation_tag=1;
    else if(operation=="subtract_exact_parent_at_final_position")node.operation_tag=2;
    else return false;
    if(node.operation_tag==2){if(attrs->dict.size()!=4||!integer(field(*attrs,"final_position"),value)||value!=final_position)return false;node.final_position=final_position;}
    else if(attrs->dict.size()!=3)return false;return true;
  }
  if(node.kernel_id=="layer_norm.v1"){if(attrs->dict.size()!=3||!integer(field(*attrs,"axis"),value)||value!=-1)return false;node.axis=-1;return true;}
  if(node.kernel_id=="pairwise_affine.v1"){
    if(attrs->dict.size()<3||attrs->dict.size()>5||!string_is(field(*attrs,"weight_layout"),"input_output"))return false;
    for(const auto& item:attrs->dict)if(item.first!="dependency_mask_spec"&&item.first!="depends_on_t"
        &&item.first!="weight_layout"&&item.first!="torch_float_kernel"&&item.first!="selected_gates")return false;
    const BinaryValue* float_kernel=field(*attrs,"torch_float_kernel");
    if(float_kernel){if(string_is(float_kernel,"batch_addmm"))node.float_kernel_tag=1;
      else if(string_is(float_kernel,"linear"))node.float_kernel_tag=2;else return false;}
    const BinaryValue* gates=field(*attrs,"selected_gates");
    if(gates){if(gates->kind!=BinaryValue::kList||gates->list.size()!=selected_gates.size())return false;
      for(std::size_t i=0;i<gates->list.size();++i)if(gates->list[i].kind!=BinaryValue::kInt
          ||gates->list[i].integer!=selected_gates[i])return false;node.selected_gates=selected_gates;}
    return true;
  }
  if(node.kernel_id=="gelu_new.v1"||node.kernel_id=="residual_add.v1")return attrs->dict.size()==2;
  if(node.kernel_id=="causal_attention.v1"){
    const BinaryValue* pivot=field(*attrs,"softmax_pivot");std::int64_t heads,head;
    if(attrs->dict.size()!=7||!integer(field(*attrs,"n_heads"),heads)||heads!=n_heads
        ||!integer(field(*attrs,"d_head"),head)||head!=d_head
        ||!string_is(field(*attrs,"mask"),"causal_delete_future")
        ||!string_is(field(*attrs,"score_scale"),"inverse_sqrt_d_head")||!pivot||pivot->kind!=BinaryValue::kDict
        ||pivot->dict.size()!=2||!string_is(field(*pivot,"kind"),"fixed_index")
        ||!integer(field(*pivot,"index"),value)||value!=0)return false;
    node.n_heads=heads;node.d_head=head;return true;
  }
  if(node.kernel_id=="final_contrast.v1"){
    std::int64_t position,width;if(attrs->dict.size()!=6||!integer(field(*attrs,"final_position"),position)
        ||position!=final_position||!integer(field(*attrs,"contrast_width"),width)||width!=contrast_width
        ||!string_is(field(*attrs,"reduction"),"fixed_balanced_pairwise")
        ||!string_is(field(*attrs,"scalarization"),"exact_affine_fusion_to_residual_contrast"))return false;
    node.final_position=position;node.contrast_width=width;return true;
  }
  if(node.kernel_id=="branch_linear_combination.v1"){
    const BinaryValue* order=field(*attrs,"order");const BinaryValue* weights=field(*attrs,"weights");
    const char* names[4]={"PAT_J","PAT_B","TAR_J","TAR_B"};const std::int64_t expected[4]={1,-1,-1,1};
    if(attrs->dict.size()!=5||!string_is(field(*attrs,"reduction"),"PAT_J_minus_PAT_B_minus_TAR_J_plus_TAR_B")
        ||!order||order->kind!=BinaryValue::kList||order->list.size()!=4
        ||!weights||weights->kind!=BinaryValue::kList||weights->list.size()!=4)return false;
    for(std::size_t i=0;i<4;++i){std::int64_t weight;if(!string_is(&order->list[i],names[i])
        ||!integer(&weights->list[i],weight)||weight!=expected[i])return false;}return true;
  }
  return false;
}

bool validate_payload_tables(const std::uint8_t* data,std::size_t size,
                             const std::uint8_t* header,std::uint32_t header_fusion,
                             std::uint64_t expected_blob_size,NativePlanTables& output){
  NativePlanTables built;
  BinaryValue root;
  if(!BinaryDecoder(data,size).parse(root)||root.kind!=BinaryValue::kDict||root.dict.size()!=21)return false;
  std::string schema,claim,blob_name;
  if(!text(field(root,"schema_version"),schema)||schema!="green-v400-native-execution-descriptor-payload-v1"
      ||!text(field(root,"claim_status"),claim)||claim!="PASS_NATIVE_DESCRIPTOR_PREPARE_ONLY"
      ||!text(field(root,"blob_name"),blob_name)||blob_name.empty()
      ||blob_name.find('/')!=std::string::npos||blob_name.find('\\')!=std::string::npos)return false;
  const BinaryValue* outcome=field(root,"contains_scientific_outcome");
  const BinaryValue* ready=field(root,"native_execution_ready");
  if(!outcome||outcome->kind!=BinaryValue::kBool||outcome->boolean
      ||!ready||ready->kind!=BinaryValue::kBool||ready->boolean)return false;
  std::int64_t blob_nbytes,alignment;
  if(!integer(field(root,"blob_nbytes"),blob_nbytes)||blob_nbytes<0
      ||static_cast<std::uint64_t>(blob_nbytes)!=expected_blob_size
      ||!integer(field(root,"alignment_bytes"),alignment)||alignment!=64)return false;
  const BinaryValue* identity=field(root,"program_execution_identity");
  const BinaryValue* fusion=field(root,"exact_final_contrast_fusion");
  if(!identity||!fusion
      ||!text_matches_digest(field(root,"program_execution_semantic_hash"),header+80)
      ||!text_matches_digest(field(root,"program_dispatch_signature_sha256"),header+112)
      ||!text_matches_digest(field(root,"blob_sha256"),header+144)
      ||!text_matches_digest(field(root,"exact_final_contrast_fusion_sha256"),header+176)
      ||!sha(field(root,"tensor_store_record_closure_sha256")))return false;

  const BinaryValue* dimensions=field(root,"dimensions");
  if(!dimensions||dimensions->kind!=BinaryValue::kDict||dimensions->dict.size()!=8)return false;
  std::int64_t dmodel,heads,dhead,sequence,final_position,dmlp,contrast_width;
  if(!integer(field(*dimensions,"d_model"),dmodel)||!integer(field(*dimensions,"n_heads"),heads)
      ||!integer(field(*dimensions,"d_head"),dhead)||!integer(field(*dimensions,"sequence_length"),sequence)
      ||!integer(field(*dimensions,"final_position"),final_position)||!integer(field(*dimensions,"d_mlp"),dmlp)
      ||!integer(field(*dimensions,"contrast_width"),contrast_width)||dmodel<=0||heads<=0||dhead<=0
      ||dmlp<=0||contrast_width<=0||dmodel!=heads*dhead||sequence<=0||final_position<0
      ||final_position>=sequence||dmodel>10000000||heads>10000000||dhead>10000000
      ||dmlp>10000000||contrast_width>10000000||sequence>10000000
      ||static_cast<std::uint64_t>(dmodel)!=header_fusion)return false;
  const BinaryValue* gates=field(*dimensions,"selected_gates");std::unordered_set<std::int64_t> selected;
  if(!gates||gates->kind!=BinaryValue::kList)return false;
  for(const auto& gate:gates->list)if(gate.kind!=BinaryValue::kInt||gate.integer<0||gate.integer>=dmlp
      ||!selected.insert(gate.integer).second)return false;
  built.sequence_length=sequence;built.d_model=dmodel;built.d_mlp=dmlp;built.n_heads=heads;
  built.d_head=dhead;built.final_position=final_position;built.contrast_width=contrast_width;
  for(const auto& gate:gates->list)built.selected_gates.push_back(static_cast<std::uint32_t>(gate.integer));

  const BinaryValue* records=field(root,"records");
  if(!records||records->kind!=BinaryValue::kList||records->list.size()!=32)return false;
  std::uint64_t prior=0;std::vector<std::string> record_names,record_semantics;
  std::unordered_set<std::string> unique_names;record_names.reserve(32);record_semantics.reserve(32);
  for(const auto& record:records->list){
    std::string name,dtype,semantic;std::int64_t offset,nbytes;
    if(record.kind!=BinaryValue::kDict||!text(field(record,"name"),name)||name.empty()
        ||!unique_names.insert(name).second||!text(field(record,"dtype"),dtype)
        ||!text(field(record,"tensor_semantic_sha256"),semantic)||!sha(field(record,"tensor_semantic_sha256"))
        ||!integer(field(record,"offset"),offset)||!integer(field(record,"nbytes"),nbytes)
        ||offset<0||nbytes<0||!sha(field(record,"data_sha256")))return false;
    const BinaryValue* shape=field(record,"shape");const std::size_t item_size=dtype_size(dtype);
    if(!shape||shape->kind!=BinaryValue::kList||shape->list.size()>8||item_size==0)return false;
    std::uint64_t elements=1;
    for(const auto& dim:shape->list){
      if(dim.kind!=BinaryValue::kInt||dim.integer<0||dim.integer>10000000)return false;
      if(dim.integer&&elements>std::numeric_limits<std::uint64_t>::max()/static_cast<std::uint64_t>(dim.integer))return false;
      elements*=static_cast<std::uint64_t>(dim.integer);
    }
    if(elements>std::numeric_limits<std::uint64_t>::max()/item_size)return false;
    const std::uint64_t expected=(prior+63U)/64U*64U;
    if(static_cast<std::uint64_t>(offset)!=expected||elements*item_size!=static_cast<std::uint64_t>(nbytes))return false;
    prior=static_cast<std::uint64_t>(offset)+static_cast<std::uint64_t>(nbytes);
    NativeRecord typed;typed.name=name;typed.dtype=dtype;typed.tensor_semantic_sha256=semantic;
    text(field(record,"data_sha256"),typed.data_sha256);typed.offset=offset;typed.nbytes=nbytes;
    for(const auto& dim:shape->list)typed.shape.push_back(static_cast<std::uint32_t>(dim.integer));
    built.records.push_back(std::move(typed));record_names.push_back(name);record_semantics.push_back(semantic);
  }
  if(prior!=expected_blob_size)return false;

  const BinaryValue* nodes=identity?field(*identity,"nodes"):nullptr;
  if(!identity||identity->kind!=BinaryValue::kDict||!nodes||nodes->kind!=BinaryValue::kList||nodes->list.size()!=81)return false;
  struct NodeInfo{std::string kernel;std::vector<std::string> tensor_hashes;std::int64_t axis0=-1;};
  std::unordered_set<std::string> seen;std::unordered_map<std::string,NodeInfo> node_info;
  std::unordered_map<std::string,std::uint32_t> node_indices;
  std::vector<std::string> ordered;ordered.reserve(81);
  for(const auto& node:nodes->list){
    std::string id,kernel;
    if(!sha(field(node,"semantic_id"))||!text(field(node,"semantic_id"),id)
        ||!text(field(node,"kernel_id"),kernel)||kernel_tag(kernel)==0||seen.count(id))return false;
    const BinaryValue* parents=field(node,"parent_semantic_ids");const BinaryValue* inputs=field(node,"tensor_inputs");
    const BinaryValue* output_spec=field(node,"output_spec");const BinaryValue* output_shape=output_spec?field(*output_spec,"shape"):nullptr;
    if(!parents||parents->kind!=BinaryValue::kList||!inputs||inputs->kind!=BinaryValue::kList
        ||!output_spec||output_spec->kind!=BinaryValue::kDict||output_spec->dict.size()!=3
        ||!output_shape||output_shape->kind!=BinaryValue::kList||output_shape->list.size()>8)return false;
    NativeNode typed;typed.semantic_id=id;typed.kernel_id=kernel;typed.kernel_tag=kernel_tag(kernel);
    if(!text(field(*output_spec,"dtype"),typed.output_dtype)||dtype_size(typed.output_dtype)==0
        ||!string_is(field(*output_spec,"layout"),"C"))return false;
    for(const auto& parent:parents->list){if(parent.kind!=BinaryValue::kString||!seen.count(parent.string))return false;
      typed.parents.push_back(node_indices[parent.string]);}
    NodeInfo info;info.kernel=kernel;
    for(const auto& input:inputs->list){std::string tensor_hash;if(!sha(field(input,"tensor_sha256"))
        ||!text(field(input,"tensor_sha256"),tensor_hash))return false;info.tensor_hashes.push_back(tensor_hash);typed.tensor_hashes.push_back(tensor_hash);}
    for(std::size_t i=0;i<output_shape->list.size();++i){const auto& dim=output_shape->list[i];
      if(dim.kind!=BinaryValue::kInt||dim.integer<0||dim.integer>10000000)return false;
      typed.output_shape.push_back(static_cast<std::uint32_t>(dim.integer));if(i==0)info.axis0=dim.integer;}
    if(!validate_node_attrs(field(node,"exact_attrs"),output_spec,static_cast<std::uint32_t>(final_position),
        static_cast<std::uint32_t>(heads),static_cast<std::uint32_t>(dhead),static_cast<std::uint32_t>(contrast_width),
        built.selected_gates,typed))return false;
    const std::uint32_t node_index=static_cast<std::uint32_t>(built.nodes.size());
    seen.insert(id);ordered.push_back(id);node_info.emplace(id,std::move(info));node_indices.emplace(id,node_index);
    built.nodes.push_back(std::move(typed));
  }
  const BinaryValue* roots=field(root,"branch_roots");
  if(!roots||roots->kind!=BinaryValue::kDict||roots->dict.size()!=4
      ||!field(*roots,"PAT_J")||!field(*roots,"PAT_B")||!field(*roots,"TAR_J")||!field(*roots,"TAR_B"))return false;
  for(const auto& item:roots->dict)if(item.second.kind!=BinaryValue::kString||!seen.count(item.second.string))return false;
  std::string output_root_id;if(!text(field(root,"output_root"),output_root_id)||!seen.count(output_root_id))return false;
  const char* root_names[4]={"PAT_J","PAT_B","TAR_J","TAR_B"};
  for(std::size_t i=0;i<4;++i){std::string id;if(!text(field(*roots,root_names[i]),id))return false;built.branch_roots[i]=node_indices[id];}
  built.output_root=node_indices[output_root_id];
  const BinaryValue* live=field(root,"required_axis0_rows");
  if(!live||live->kind!=BinaryValue::kList||live->list.size()!=81)return false;
  for(std::size_t i=0;i<81;++i){
    std::string id;if(!text(field(live->list[i],"node_semantic_id"),id)||id!=ordered[i])return false;
    const BinaryValue* rows=field(live->list[i],"rows");if(!rows||rows->kind!=BinaryValue::kList)return false;
    std::int64_t prior_row=-1;const std::int64_t axis0=node_info[id].axis0;
    for(const auto& row:rows->list){if(row.kind!=BinaryValue::kInt||row.integer<=prior_row
        ||axis0<0||row.integer<0||row.integer>=axis0)return false;prior_row=row.integer;
      built.nodes[i].required_axis0_rows.push_back(static_cast<std::uint32_t>(row.integer));++built.total_liveness_rows;}
  }

  const BinaryValue* weights=fusion?field(*fusion,"weights"):nullptr;
  const BinaryValue* closure=fusion?field(*fusion,"input_closure"):nullptr;std::int64_t fusion_dmodel;
  if(!fusion||fusion->kind!=BinaryValue::kDict||!weights||weights->kind!=BinaryValue::kList
      ||weights->list.size()!=header_fusion||!integer(field(*fusion,"d_model"),fusion_dmodel)
      ||fusion_dmodel!=dmodel||!closure||closure->kind!=BinaryValue::kDict)return false;
  for(const auto& weight:weights->list){NativeDyadic typed;if(!dyadic(&weight,typed))return false;built.fusion_weights.push_back(std::move(typed));}
  if(!dyadic(field(*fusion,"bias"),built.fusion_bias))return false;
  std::unordered_map<std::string,std::string> fusion_semantics;std::unordered_map<std::string,std::uint32_t> fusion_indices;
  for(const auto& item:closure->dict){std::string semantic;if(!sha(field(item.second,"semantic_sha256"))
      ||!text(field(item.second,"semantic_sha256"),semantic))return false;
    fusion_indices.emplace(item.first,static_cast<std::uint32_t>(fusion_indices.size()));fusion_semantics.emplace(item.first,semantic);}

  const BinaryValue* bindings=field(root,"program_input_binding_table");
  if(!bindings||bindings->kind!=BinaryValue::kList||bindings->list.size()!=150)return false;
  std::size_t binding_index=0;
  for(const std::string& expected_id:ordered){const NodeInfo& info=node_info[expected_id];
    for(std::size_t expected_ordinal=0;expected_ordinal<info.tensor_hashes.size();++expected_ordinal){
      if(binding_index>=bindings->list.size())return false;const BinaryValue& binding=bindings->list[binding_index++];
      std::string id,kernel,tensor_hash;std::int64_t ordinal;
      if(!text(field(binding,"node_semantic_id"),id)||id!=expected_id
          ||!text(field(binding,"kernel_id"),kernel)||kernel!=info.kernel
          ||!integer(field(binding,"tensor_input_ordinal"),ordinal)||ordinal!=static_cast<std::int64_t>(expected_ordinal)
          ||!text(field(binding,"tensor_semantic_sha256"),tensor_hash)||tensor_hash!=info.tensor_hashes[expected_ordinal])return false;
      const BinaryValue* source=field(binding,"source");std::string kind;if(!source||!text(field(*source,"kind"),kind))return false;
      if(kind=="packed_record"){
        std::int64_t index;std::string name;if(!integer(field(*source,"record_index"),index)||index<0||index>=32
            ||!text(field(*source,"record_name"),name)||name!=record_names[index]
            ||tensor_hash!=record_semantics[index])return false;
        NativeBinding typed;typed.node_index=node_indices[id];typed.input_ordinal=ordinal;typed.source_is_record=true;
        typed.source_index=index;typed.tensor_semantic_sha256=tensor_hash;typed.source_name=name;built.bindings.push_back(std::move(typed));
      }else if(kind=="exact_final_contrast_fusion_source"){
        std::string source_name;if(!text(field(*source,"source_name"),source_name)||!fusion_semantics.count(source_name)
            ||tensor_hash!=fusion_semantics[source_name])return false;
        NativeBinding typed;typed.node_index=node_indices[id];typed.input_ordinal=ordinal;typed.source_is_record=false;
        typed.source_index=fusion_indices[source_name];typed.tensor_semantic_sha256=tensor_hash;typed.source_name=source_name;
        built.bindings.push_back(std::move(typed));
      }else return false;
    }
  }
  if(binding_index!=bindings->list.size())return false;

  const BinaryValue* policy=field(root,"native_runtime_policy");std::int64_t rc,nc,bc,version,events;
  std::string policy_schema,abi,rounding,domain,locator;
  const BinaryValue* fallback=policy?field(*policy,"corruption_fallback_allowed"):nullptr;
  const BinaryValue* precisions=policy?field(*policy,"supported_precision_bits"):nullptr;
  if(!policy||policy->kind!=BinaryValue::kDict||policy->dict.size()!=12
      ||!text(field(*policy,"schema_version"),policy_schema)||policy_schema!="green-v400-native-runtime-policy-v1"
      ||!integer(field(*policy,"descriptor_format_version"),version)||version!=1
      ||!text(field(*policy,"compiled_kernel_abi"),abi)||abi!="green-v400-compiled-mpfr-v2"
      ||!text(field(*policy,"rounding_contract"),rounding)||rounding!="directed-mpfr-outward-interval-jet2"
      ||!text(field(*policy,"domain_schema"),domain)||domain!="closed-dyadic-interval-v1"
      ||!text(field(*policy,"blob_locator_policy"),locator)||locator!="explicit-path-plus-nbytes-sha256-record-closure"
      ||!integer(field(*policy,"exact_successful_dispatch_event_count"),events)||events!=81
      ||!integer(field(*policy,"record_count"),rc)||!integer(field(*policy,"node_count"),nc)
      ||!integer(field(*policy,"binding_count"),bc)||rc!=32||nc!=81||bc!=150
      ||!fallback||fallback->kind!=BinaryValue::kBool||fallback->boolean
      ||!precisions||precisions->kind!=BinaryValue::kList||precisions->list.size()!=2
      ||precisions->list[0].kind!=BinaryValue::kInt||precisions->list[0].integer!=384
      ||precisions->list[1].kind!=BinaryValue::kInt||precisions->list[1].integer!=512)return false;
  output=std::move(built);return true;
}

struct PlanEnvelope { int dfd=-1,bfd=-1; void* blob=MAP_FAILED; std::uint64_t dsize=0,bsize=0; std::uint32_t records=0,nodes=0,bindings=0,fusion=0; bool payload_validated=false; NativePlanTables tables; ~PlanEnvelope(){if(blob!=MAP_FAILED)::munmap(blob,bsize);if(dfd>=0)::close(dfd);if(bfd>=0)::close(bfd);} };
struct NativePrecisionContext {
  std::shared_ptr<PlanEnvelope> plan;
  std::uint32_t precision=0;
  std::vector<void*> static_buffers;
  std::vector<std::uint32_t> static_record_indices;
  std::uint64_t static_jet_count=0;
  ~NativePrecisionContext(){for(void* buffer:static_buffers)green_v400_resident_jet_buffer_free(buffer);}
};
std::mutex registry_mutex;std::unordered_map<std::uint64_t,std::shared_ptr<PlanEnvelope>> registry;
std::atomic<std::uint64_t> next_handle{1};
std::mutex context_registry_mutex;std::unordered_map<std::uint64_t,std::unique_ptr<NativePrecisionContext>> context_registry;
std::atomic<std::uint64_t> next_context_handle{1};

}  // namespace

extern "C" int green_v400_native_plan_envelope_open_v1(
    const char* descriptor_path,const char* blob_path,const char* expected_descriptor_sha,
    const char* expected_program_sha,const char* expected_dispatch_sha,const char* expected_blob_sha,
    const char* expected_fusion_sha,std::uint64_t expected_blob_nbytes,
    std::uint32_t expected_fusion_weights,std::uint64_t* out_handle) {
  if(!descriptor_path||!blob_path||!out_handle)return 2;*out_handle=0;
  std::array<std::uint8_t,32> ed,ep,et,eb,ef;
  if(!parse_hex(expected_descriptor_sha,ed)||!parse_hex(expected_program_sha,ep)||!parse_hex(expected_dispatch_sha,et)||!parse_hex(expected_blob_sha,eb)||!parse_hex(expected_fusion_sha,ef))return 2;
  std::shared_ptr<PlanEnvelope> plan(new PlanEnvelope());
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
  if(!validate_payload_tables(bytes.data()+256,bytes.size()-256,h,plan->fusion,plan->bsize,plan->tables))return 11;plan->payload_validated=true;
  Sha256 blob_hash;std::vector<std::uint8_t> chunk(1U<<20);std::uint64_t off=0;while(off<plan->bsize){std::size_t want=std::min<std::uint64_t>(chunk.size(),plan->bsize-off);ssize_t n=::pread(plan->bfd,chunk.data(),want,off);if(n<=0)return 8;blob_hash.update(chunk.data(),n);off+=n;}if(blob_hash.finish()!=eb)return 8;
  plan->blob=::mmap(nullptr,plan->bsize,PROT_READ,MAP_PRIVATE,plan->bfd,0);if(plan->blob==MAP_FAILED)return 9;
  std::uint64_t handle=next_handle.fetch_add(1);if(handle==0)return 10;{std::lock_guard<std::mutex> lock(registry_mutex);registry.emplace(handle,plan);}*out_handle=handle;return 0;
}

extern "C" int green_v400_native_plan_envelope_info_v1(std::uint64_t handle,std::uint64_t* descriptor_nbytes,std::uint64_t* blob_nbytes,std::uint32_t* records,std::uint32_t* nodes,std::uint32_t* bindings,std::uint32_t* fusion_weights){std::lock_guard<std::mutex> lock(registry_mutex);auto it=registry.find(handle);if(it==registry.end())return 2;auto& p=*it->second;if(descriptor_nbytes)*descriptor_nbytes=p.dsize;if(blob_nbytes)*blob_nbytes=p.bsize;if(records)*records=p.records;if(nodes)*nodes=p.nodes;if(bindings)*bindings=p.bindings;if(fusion_weights)*fusion_weights=p.fusion;return 0;}
extern "C" int green_v400_native_plan_envelope_close_v1(std::uint64_t handle){std::shared_ptr<PlanEnvelope> removed;{std::lock_guard<std::mutex> lock(registry_mutex);auto it=registry.find(handle);if(it==registry.end())return 2;removed=std::move(it->second);registry.erase(it);}return 0;}
extern "C" int green_v400_native_plan_payload_validated_v1(std::uint64_t handle){std::lock_guard<std::mutex> lock(registry_mutex);auto it=registry.find(handle);if(it==registry.end())return 0;return it->second->payload_validated?1:0;}
extern "C" int green_v400_native_plan_typed_info_v1(std::uint64_t handle,std::uint32_t* records,std::uint32_t* nodes,std::uint32_t* bindings,std::uint32_t* fusion_weights,std::uint64_t* liveness_rows,std::uint32_t* root_count){
  std::lock_guard<std::mutex> lock(registry_mutex);auto it=registry.find(handle);if(it==registry.end())return 2;
  const auto& tables=it->second->tables;if(records)*records=tables.records.size();if(nodes)*nodes=tables.nodes.size();
  if(bindings)*bindings=tables.bindings.size();if(fusion_weights)*fusion_weights=tables.fusion_weights.size();
  if(liveness_rows)*liveness_rows=tables.total_liveness_rows;if(root_count)*root_count=tables.branch_roots.size();return 0;
}
extern "C" int green_v400_native_plan_typed_trace_v1(std::uint64_t handle,std::uint32_t* kernel_tags,std::uint32_t* liveness_counts,std::uint32_t node_capacity,std::uint32_t* roots,std::uint32_t root_capacity,std::uint32_t* output_root){
  std::lock_guard<std::mutex> lock(registry_mutex);auto it=registry.find(handle);if(it==registry.end())return 2;
  const auto& tables=it->second->tables;if(!kernel_tags||!liveness_counts||node_capacity<tables.nodes.size()
      ||!roots||root_capacity<tables.branch_roots.size()||!output_root)return 3;
  for(std::size_t i=0;i<tables.nodes.size();++i){kernel_tags[i]=tables.nodes[i].kernel_tag;
    liveness_counts[i]=tables.nodes[i].required_axis0_rows.size();}
  for(std::size_t i=0;i<tables.branch_roots.size();++i)roots[i]=tables.branch_roots[i];
  *output_root=tables.output_root;return 0;
}
extern "C" int green_v400_native_precision_context_open_v1(std::uint64_t plan_handle,std::uint32_t precision_bits,std::uint64_t* out_context_handle){
  if(!out_context_handle||(precision_bits!=384&&precision_bits!=512))return 2;*out_context_handle=0;
  std::shared_ptr<PlanEnvelope> plan;{std::lock_guard<std::mutex> lock(registry_mutex);auto it=registry.find(plan_handle);
    if(it==registry.end())return 2;plan=it->second;}
  const std::uint32_t endian_probe=1;if(*reinterpret_cast<const std::uint8_t*>(&endian_probe)!=1)return 3;
  std::unique_ptr<NativePrecisionContext> context(new NativePrecisionContext());context->plan=plan;context->precision=precision_bits;
  const char* static_names[5]={"zero.d_model","PAT.resid_mid","PAT.resid_post","TAR.resid_mid","TAR.resid_post"};
  for(const char* name:static_names){
    std::size_t index=0;while(index<plan->tables.records.size()&&plan->tables.records[index].name!=name)++index;
    if(index==plan->tables.records.size())return 3;const NativeRecord& record=plan->tables.records[index];
    if(record.dtype!="<f4"||record.nbytes==0||record.nbytes%4!=0||record.nbytes/4>1000000U
        ||record.offset>plan->bsize||record.nbytes>plan->bsize-record.offset)return 3;
    const auto* bits=reinterpret_cast<const std::uint32_t*>(
        static_cast<const std::uint8_t*>(plan->blob)+record.offset);void* buffer=nullptr;
    const int status=green_v400_resident_jet_buffer_import_f32_constants(
        precision_bits,static_cast<std::uint32_t>(record.nbytes/4),bits,&buffer);
    if(status!=0||!buffer)return 12;context->static_buffers.push_back(buffer);
    context->static_record_indices.push_back(static_cast<std::uint32_t>(index));context->static_jet_count+=record.nbytes/4;
  }
  const std::uint64_t handle=next_context_handle.fetch_add(1);if(handle==0)return 10;
  {std::lock_guard<std::mutex> lock(context_registry_mutex);context_registry.emplace(handle,std::move(context));}
  *out_context_handle=handle;return 0;
}
extern "C" int green_v400_native_precision_context_info_v1(std::uint64_t context_handle,std::uint32_t* precision_bits,std::uint32_t* static_buffer_count,std::uint64_t* static_jet_count,std::uint32_t* node_count,std::uint32_t* binding_count){
  std::lock_guard<std::mutex> lock(context_registry_mutex);auto it=context_registry.find(context_handle);if(it==context_registry.end())return 2;
  const auto& context=*it->second;if(precision_bits)*precision_bits=context.precision;
  if(static_buffer_count)*static_buffer_count=context.static_buffers.size();if(static_jet_count)*static_jet_count=context.static_jet_count;
  if(node_count)*node_count=context.plan->tables.nodes.size();if(binding_count)*binding_count=context.plan->tables.bindings.size();return 0;
}
extern "C" int green_v400_native_precision_context_close_v1(std::uint64_t context_handle){
  std::unique_ptr<NativePrecisionContext> removed;{std::lock_guard<std::mutex> lock(context_registry_mutex);
    auto it=context_registry.find(context_handle);if(it==context_registry.end())return 2;removed=std::move(it->second);context_registry.erase(it);}return 0;
}
