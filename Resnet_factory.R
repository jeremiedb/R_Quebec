
# input
data <- mx.symbol.Variable('data')

conv_factory <- function(data, num_filter, kernel, stride,
                         pad, act_type = 'relu', conv_type = 0) {
  if (conv_type == 0) {
    conv = mx.symbol.Convolution(data = data, num_filter = num_filter,
                                 kernel = kernel, stride = stride, pad = pad)
    bn = mx.symbol.BatchNorm(data = conv)
    act = mx.symbol.Activation(data = bn, act_type = act_type)
    return(act)
  } else if (conv_type == 1) {
    conv = mx.symbol.Convolution(data = data, num_filter = num_filter,
                                 kernel = kernel, stride = stride, pad = pad)
    bn = mx.symbol.BatchNorm(data = conv)
    return(bn)
  }
}

residual_factory <- function(data, num_filter, dim_match) {
  if (dim_match) {
    identity_data = data
    conv1 = conv_factory(data = data, num_filter = num_filter, kernel = c(3, 3),
                         stride = c(1, 1), pad = c(1, 1), act_type = 'relu', conv_type = 0)
    
    conv2 = conv_factory(data = conv1, num_filter = num_filter, kernel = c(3, 3),
                         stride = c(1, 1), pad = c(1, 1), conv_type = 1)
    new_data = identity_data + conv2
    return(new_data)
  } else {
    conv1 = conv_factory(data = data, num_filter = num_filter, kernel = c(3, 3),
                         stride = c(2, 2), pad = c(1, 1), act_type = 'relu', conv_type = 0)
    conv2 = conv_factory(data = conv1, num_filter = num_filter, kernel = c(3, 3),
                         stride = c(1, 1), pad = c(1, 1), conv_type = 1)
    
    # adopt project method in the paper when dimension increased
    project_data = conv_factory(data = data, num_filter = num_filter, kernel = c(1, 1),
                                stride = c(2, 2), pad = c(0, 0), conv_type = 1)
    new_data = project_data + conv2
    return(new_data)
  }
}

residual_net <- function(data, n) {
  #fisrt 2n layers
  for (i in 1:n) {
    data = residual_factory(data = data, num_filter = 16, dim_match = TRUE)
  }
  
  #second 2n layers
  for (i in 1:n) {
    if (i == 1) {
      data = residual_factory(data = data, num_filter = 32, dim_match = FALSE)
    } else {
      data = residual_factory(data = data, num_filter = 32, dim_match = TRUE)
    }
  }
  #third 2n layers
  for (i in 1:n) {
    if (i == 1) {
      data = residual_factory(data = data, num_filter = 64, dim_match = FALSE)
    } else {
      data = residual_factory(data = data, num_filter = 64, dim_match = TRUE)
    }
  }
  #fourth 2n layers
  for (i in 1:n) {
    if (i == 1) {
      data = residual_factory(data = data, num_filter = 128, dim_match = FALSE)
    } else {
      data = residual_factory(data = data, num_filter = 128, dim_match = TRUE)
    }
  }
  return(data)
}

Resnet_factory <- function(num_classes = 10, loops) {
  conv <- conv_factory(data = mx.symbol.Variable(name = 'data'), num_filter = 16,
                       kernel = c(5, 5), stride = c(2, 2), pad = c(2, 2),
                       act_type = 'relu', conv_type = 0)
  n <- loops # set n = 3 means get a model with 3*6+2=20 layers, set n = 9 means 9*6+2=56 layers
  resnet <- residual_net(conv, n) #
  pool <- mx.symbol.Pooling(data = resnet, kernel = c(7, 7), pool_type = 'avg')
  flatten <- mx.symbol.Flatten(data = pool, name = 'flatten')
  fc <- mx.symbol.FullyConnected(data = flatten, num_hidden = num_classes, name = 'fc1')
  softmax <- mx.symbol.SoftmaxOutput(data = fc, name = 'softmax')
  return(softmax)
}
