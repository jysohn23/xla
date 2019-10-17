#include "tensorflow/compiler/xla/xla_client/computation_client.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/mesh_service.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace xla {
namespace {

struct Device {
  string kind;
  int ordinal = 0;
};

Device ParseDevice(const string& device) {
  std::vector<string> parts = absl::StrSplit(device, ':');
  XLA_CHECK_EQ(parts.size(), 2) << device;
  return {parts[0], std::stoi(parts[1])};
}

ComputationClient* CreateClient() {
  return ComputationClient::Create().release();
}

XrtComputationClient::Worker ParseWorker(const string& worker) {
  std::vector<string> parts = absl::StrSplit(worker, ':');
  XLA_CHECK(parts.size() == 1 || parts.size() == 2) << worker;
  return parts.size() == 1
             ? XrtComputationClient::Worker(parts[0], 0)
             : XrtComputationClient::Worker(parts[0], std::stoi(parts[1]));
}

string MakeGrpcEndPoint(const string& server) {
  return server.compare(0, 7, "grpc://") == 0 ? server
                                              : absl::StrCat("grpc://", server);
}

string GetXrtDevicePath(const string& worker, int task_no,
                        const string& device_type, int ordinal) {
  return absl::StrCat("/job:", worker, "/replica:0/task:", task_no,
                      "/device:", device_type, ":", ordinal);
}

bool IsLocalDevice(
    const XrtComputationClient::Worker& worker,
    const tensorflow::DeviceNameUtils::ParsedName& parsed_device) {
  if (worker.name != parsed_device.job ||
      worker.task_no != parsed_device.task) {
    return false;
  }
  string mp_device = XrtComputationClient::GetMultiProcessingDevice();
  if (mp_device.empty()) {
    // We never hit this branch in any case
    TF_VLOG(3) << "IsLocalDevice mp_device: EMPTY";
    return true;
  } else {
    TF_VLOG(3) << "IsLocalDevice mp_device: " << mp_device;
  }
  // ex. mp_device = 'TPU:16'
  Device device = ParseDevice(mp_device);
  // ex. device.type = 'TPU', device.id = 16
  TF_VLOG(3) << "IsLocalDevice compare_mp: "
             << "device.ordinal=" << device.ordinal << "|"
             << "device.kind=" << device.kind << ", "
             << "parsed_device.id=" << parsed_device.id << "|"
             << "parsed_device.type=" << parsed_device.type;
  // This comparison always returns false when on non-0 host
  const int devices_per_worker = sys_util::GetEnvInt("TPU_NUM_DEVICES", 8);
  return device.ordinal ==
             (devices_per_worker * parsed_device.task + parsed_device.id) &&
         device.kind == parsed_device.type;
}

void PopulateLocalDevices(XrtComputationClient::Options* options) {
  string local_worker = sys_util::GetEnvString("XRT_LOCAL_WORKER", "");
  TF_VLOG(3) << "local_worker: " << local_worker;
  XrtComputationClient::Worker worker("", -1);
  if (!local_worker.empty()) {
    worker = ParseWorker(local_worker);
  }
  TF_VLOG(3) << "Worker: " << worker.name << ", " << worker.task_no;
  std::map<string, int> min_ordinals;

  TF_VLOG(3) << "Options.global_device_map: {";
  for (const auto& dev_target : options->global_device_map) {
    TF_VLOG(3) << "\t" << dev_target.first << ": " << dev_target.second;
  }

  for (auto& device_xrt_device : options->global_device_map) {
    if (worker.task_no >= 0) {
      tensorflow::DeviceNameUtils::ParsedName parsed_device;
      XLA_CHECK(tensorflow::DeviceNameUtils::ParseFullName(
                    device_xrt_device.second, &parsed_device) &&
                parsed_device.has_job && parsed_device.has_task &&
                parsed_device.has_id && parsed_device.has_type)
          << device_xrt_device.second;
      if (!IsLocalDevice(worker, parsed_device)) {
        // 100% of the time of non-0 host we hit this branch so we never even
        // find
        TF_VLOG(3) << "Is NOT LocalDevice(worker=" << worker.name << "|"
                   << worker.task_no << ", "
                   << "parsed_device=" << parsed_device.job << "|"
                   << parsed_device.task << "|" << parsed_device.id << "|"
                   << parsed_device.type << ")";
        continue;
      } else {
        TF_VLOG(3) << "ISLocalDevice(worker=" << worker.name << "|"
                   << worker.task_no << ", "
                   << "parsed_device=" << parsed_device.job << "|"
                   << parsed_device.task << "|" << parsed_device.id << "|"
                   << parsed_device.type << ")";
      }
    }
    options->devices.insert(device_xrt_device.first);

    Device global_device = ParseDevice(device_xrt_device.first);
    auto it = min_ordinals.find(global_device.kind);
    TF_VLOG(3) << "finding in min_ordinals global_device.kind: " << global_device.kind;
    if (it == min_ordinals.end()) {
      TF_VLOG(3) << "it == min_ordinals.end() emplacing..";
      min_ordinals.emplace(global_device.kind, global_device.ordinal);
    } else {
      TF_VLOG(3) << "it != min_ordinals.end() ";
      it->second = std::min<int>(it->second, global_device.ordinal);
      // TODO: maybe we need to emplace here too?
    }
  }
  // min_ordinals empty for non 0 hosts.
  TF_VLOG(3) << "min_ordinals after: {";
  for (auto& ordinal : min_ordinals) {
    TF_VLOG(3) << "\t" << ordinal.first << ": " << ordinal.second;
  }
  TF_VLOG(3) << "}";
  for (auto kind : {"TPU", "GPU", "CPU"}) {
    auto it = min_ordinals.find(kind);
    if (it != min_ordinals.end()) {
      options->default_device = absl::StrCat(kind, ":", it->second);
      break;
    } else {
      // This branch is consistently hit for non 0 hosts.
      TF_VLOG(3) << "debug => did not find device kind: " << kind;
    }
  }
}

void AddXrtHostDevices(const string& worker_name, int task_no,
                       const string& server,
                       std::map<string, int>* device_ordinals,
                       XrtComputationClient::Options* options) {
  struct Devices {
    const char* name;
    const char* tf_name;
    int count;
  } const devices[] = {
      {"TPU", "TPU", sys_util::GetEnvInt("TPU_NUM_DEVICES", 8)},
      {"CPU", "XLA_CPU", sys_util::GetEnvInt("CPU_NUM_DEVICES", 1)},
  };
  options->workers_map.emplace(
      XrtComputationClient::Worker(worker_name, task_no),
      MakeGrpcEndPoint(server));
  for (auto& device : devices) {
    int& device_ordinal = (*device_ordinals)[device.name];
    for (int j = 0; j < device.count; ++j, ++device_ordinal) {
      string device_name = absl::StrCat(device.name, ":", device_ordinal);
      string xrt_device_name =
          GetXrtDevicePath(worker_name, task_no, device.tf_name, j);
      options->global_device_map.emplace(device_name, xrt_device_name);
    }
  }
}

bool ParseEnvBasedTpuClusterConfig(XrtComputationClient::Options* options) {
  string tpu_config = sys_util::GetEnvString("XRT_TPU_CONFIG", "");
  if (tpu_config.empty()) {
    return false;
  }
  std::map<string, int> device_ordinals;
  std::vector<string> spec_parts = absl::StrSplit(tpu_config, '|');
  XLA_CHECK(!spec_parts.empty()) << tpu_config;
  for (const auto& spec : spec_parts) {
    std::vector<string> host_parts = absl::StrSplit(spec, ';');
    XLA_CHECK_EQ(host_parts.size(), 3) << spec;
    AddXrtHostDevices(host_parts[0], std::stoi(host_parts[1]), host_parts[2],
                      &device_ordinals, options);
  }
  return true;
}

bool ParseMeshConfig(
    XrtComputationClient::Options* options,
    std::unique_ptr<tensorflow::tpu::TopologyProto>* topology_proto) {
  service::MeshClient* client = service::MeshClient::Get();
  if (client == nullptr) {
    return false;
  }
  string local_worker_env = sys_util::GetEnvString("XRT_LOCAL_WORKER", "");
  XLA_CHECK(!local_worker_env.empty())
      << "In a mesh client setup the XRT_LOCAL_WORKER must be specified";

  XrtComputationClient::Worker local_worker = ParseWorker(local_worker_env);

  TF_LOG(INFO) << "Fetching mesh configuration for worker " << local_worker.name
               << ":" << local_worker.task_no << " from mesh service at "
               << client->address();
  service::grpc::Config config = client->GetConfig();
  TF_VLOG(3) << "Mesh Config: " << config.DebugString();

  string mp_device = XrtComputationClient::GetMultiProcessingDevice();
  for (auto& config_worker : config.workers()) {
    XrtComputationClient::Worker worker(config_worker.name(),
                                        config_worker.task_no());
    options->workers_map.emplace(worker, config_worker.address());

    for (auto& device : config_worker.devices()) {
      Device local_device = ParseDevice(device.local_name());
      options->global_device_map.emplace(
          device.global_name(),
          GetXrtDevicePath(worker.name, worker.task_no, local_device.kind,
                           local_device.ordinal));
      if (local_worker == worker &&
          (mp_device.empty() || device.global_name() == mp_device)) {
        options->devices.insert(device.global_name());
      }
    }
  }
  (*topology_proto) = absl::make_unique<tensorflow::tpu::TopologyProto>(
      std::move(*config.mutable_proto()));
  return true;
}

}  // namespace

std::unique_ptr<ComputationClient> ComputationClient::Create() {
  XrtComputationClient::Options options;
  std::unique_ptr<tensorflow::tpu::TopologyProto> topology_proto;
  if (!ParseEnvBasedTpuClusterConfig(&options) &&
      !ParseMeshConfig(&options, &topology_proto)) {
    string device_spec = sys_util::GetEnvString(
        "XRT_DEVICE_MAP",
        "TPU:0;/job:tpu_worker/replica:0/task:0/device:TPU:0");
    for (const auto& device_target : absl::StrSplit(device_spec, '|')) {
      std::vector<string> parts = absl::StrSplit(device_target, ';');
      XLA_CHECK_EQ(parts.size(), 2) << device_target;
      if (options.default_device.empty()) {
        options.default_device = parts[0];
      }
      options.global_device_map.emplace(parts[0], parts[1]);
    }
    string workers_spec = sys_util::GetEnvString(
        "XRT_WORKERS", "tpu_worker:0;grpc://localhost:51000");
    for (const auto& name_target : absl::StrSplit(workers_spec, '|')) {
      std::vector<string> parts = absl::StrSplit(name_target, ';');
      XLA_CHECK_EQ(parts.size(), 2) << name_target;
      options.workers_map.emplace(ParseWorker(parts[0]),
                                  MakeGrpcEndPoint(parts[1]));
    }
  }
  // There is a problem somewhere in PopulateLocalDevices where the
  // options.default_device doesn't get populated for n>0 local workers/hosts.
  PopulateLocalDevices(&options);
  return std::unique_ptr<ComputationClient>(
      new XrtComputationClient(options, std::move(topology_proto)));
}

std::shared_ptr<ComputationClient::Computation> ComputationClient::Compile(
    XlaComputation computation, string compilation_device,
    std::vector<string> devices, const Shape* output_shape) {
  std::vector<CompileInstance> instances;
  instances.emplace_back(std::move(computation), std::move(compilation_device),
                         std::move(devices), output_shape);
  std::vector<std::shared_ptr<Computation>> results =
      Compile(std::move(instances));
  return std::move(results[0]);
}

std::vector<std::string> ComputationClient::GetCompilationDevices(
    const std::string& device,
    tensorflow::gtl::ArraySlice<const std::string> devices) const {
  std::vector<std::string> compilation_devices;
  if (devices.empty()) {
    auto& replication_devices = GetReplicationDevices();
    if (replication_devices.empty()) {
      compilation_devices.push_back(device);
    } else {
      compilation_devices = replication_devices;
    }
  } else {
    compilation_devices.insert(compilation_devices.end(), devices.begin(),
                               devices.end());
  }
  return compilation_devices;
}

int64 ComputationClient::GetDeviceOrdinal(const string& device) {
  auto pos = device.rfind(':');
  XLA_CHECK_NE(pos, string::npos) << device;
  return std::stoi(device.substr(pos + 1));
}

ComputationClient* ComputationClient::Get() {
  static ComputationClient* computation_client = CreateClient();
  return computation_client;
}

metrics::Metric* ComputationClient::TransferToServerMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("TransferToServerTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::TransferFromServerMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("TransferFromServerTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::CompileMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("CompileTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::ExecuteMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("ExecuteTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::ExecuteReplicatedMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("ExecuteReplicatedTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::ExecuteParallelMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("ExecuteParallelTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::ExecuteChainedMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("ExecuteChainedTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::DeconstructTupleMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("DeconstructTupleTime", metrics::MetricFnTime);
  return metric;
}

metrics::Counter* ComputationClient::CreateDataHandlesCounter() {
  // Do not change the name of the counter as xla_model.py references it.
  static metrics::Counter* counter = new metrics::Counter("CreateDataHandles");
  return counter;
}

metrics::Counter* ComputationClient::ReleaseDataHandlesCounter() {
  // Do not change the name of the counter as xla_model.py references it.
  static metrics::Counter* counter = new metrics::Counter("ReleaseDataHandles");
  return counter;
}

metrics::Counter* ComputationClient::DestroyDataHandlesCounter() {
  // Do not change the name of the counter as xla_model.py references it.
  static metrics::Counter* counter = new metrics::Counter("DestroyDataHandles");
  return counter;
}

metrics::Metric* ComputationClient::ReleaseDataHandlesTimeMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("ReleaseDataHandlesTime", metrics::MetricFnTime);
  return metric;
}

metrics::Counter* ComputationClient::CreateCompileHandlesCounter() {
  // Do not change the name of the counter as xla_model.py references it.
  static metrics::Counter* counter =
      new metrics::Counter("CreateCompileHandles");
  return counter;
}

metrics::Counter* ComputationClient::ReleaseCompileHandlesCounter() {
  // Do not change the name of the counter as xla_model.py references it.
  static metrics::Counter* counter =
      new metrics::Counter("ReleaseCompileHandles");
  return counter;
}

metrics::Counter* ComputationClient::DestroyCompileHandlesCounter() {
  // Do not change the name of the counter as xla_model.py references it.
  static metrics::Counter* counter =
      new metrics::Counter("DestroyCompileHandles");
  return counter;
}

metrics::Metric* ComputationClient::ReleaseCompileHandlesTimeMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("ReleaseCompileHandlesTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::InboundDataMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("InboundData", metrics::MetricFnBytes);
  return metric;
}

metrics::Metric* ComputationClient::OutboundDataMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("OutboundData", metrics::MetricFnBytes);
  return metric;
}

}  // namespace xla
