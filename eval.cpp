
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {

  // Deserialize the ScriptModule from a file using torch::jit::load().
  torch::jit::script::Module module;
  try {
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  //Note that hist and nbrs trajecories should be computed wrt to the reference vehicle
  //The following are Dummy variables for testing
  //hist, nbrs, mask, lat_enc, lon_enc
  torch::Tensor hist = torch::zeros({16,1,2});
  torch::Tensor nbrs = torch::zeros({16,39,2});
  torch::Tensor lat_enc = torch::zeros({1,3});
  torch::Tensor lon_enc = torch::zeros({1,2});
  torch::Tensor mask = torch::ones({1,3,13,64}, {torch::kByte});


  //Concatentae the inputs into IValue Struct
  //hist, nbrs, mask, lat_enc, lon_enc
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(hist);
  inputs.push_back(nbrs);
  inputs.push_back(mask);
  inputs.push_back(lat_enc);
  inputs.push_back(lon_enc);

  //Forward Pass to evaluate the model on the given inputs
  //Out concatenates the outputs: fut_pred, lat_pred, lon_pred
  auto out = module.forward(inputs).toTuple();

  //unpack out into fut_pred, lat_pred, lon_pred
  auto fut_pred = out->elements()[0].toTensorList();
  torch::Tensor lat_pred = out->elements()[1].toTensor();
  torch::Tensor lon_pred = out->elements()[2].toTensor();

  //Find the manuever-based future trajectory
  int lat_man = lat_pred.argmax(1).item().toInt();
  int lon_man = lon_pred.argmax(1).item().toInt();
  int indx = lon_man*3 + lat_man;
  torch::Tensor fut_pred_max = fut_pred[indx];

  //Confirm
  std::cout << lat_pred << std::endl;
  std::cout << lat_man << std::endl;
  std::cout << lon_pred << std::endl;
  std::cout << lon_man << std::endl;
  std::cout << indx << std::endl;
  std::cout << fut_pred_max << std::endl;
}

