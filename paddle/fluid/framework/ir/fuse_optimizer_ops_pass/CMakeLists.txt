cc_library(
  fuse_optimizer_op_pass
  SRCS fuse_optimizer_op_pass.cc
  DEPS graph graph_helper)
cc_library(
  fuse_adam_op_pass
  SRCS fuse_adam_op_pass.cc
  DEPS fuse_optimizer_op_pass)
cc_library(
  fuse_sgd_op_pass
  SRCS fuse_sgd_op_pass.cc
  DEPS fuse_optimizer_op_pass)
cc_library(
  fuse_momentum_op_pass
  SRCS fuse_momentum_op_pass.cc
  DEPS fuse_optimizer_op_pass)
