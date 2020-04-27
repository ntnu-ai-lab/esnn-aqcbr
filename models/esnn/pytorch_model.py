import torch

class ESNNModel(torch.nn.Module):
    def __init__(self, input_shape, output_shape, networklayers=[13, 13], dropoutrate=0.052):
        """

        """
        super(ESNNModel, self).__init__()

        g_layers = networklayers
        c_layers = networklayers
        #self.register_backward_hook(self.printgradnorm)
        if isinstance(networklayers[0], list):
            # this means user has specified different layers
            # for g and c....
            g_layers = networklayers[0]
            c_layers = networklayers[1]


        # input1 = Input(shape=(X.shape[1],), dtype="float32")
        # input2 = Input(shape=(X.shape[1],), dtype="float32")

        # given S(x,y) = C(G(x),G(y)):
        # \hat{x} = G(x)
        self.G = torch.nn.ModuleList()
        self.G_size = len(g_layers)
        for networklayer in g_layers:
            self.G.append(torch.nn.Linear(in_features=input_shape,
                                          out_features=networklayer))
            self.G.append(torch.nn.Dropout(dropoutrate))
            input_shape = networklayer

        self.inner_output = torch.nn.Linear(in_features=input_shape,
                                            out_features=output_shape)

        self.C = torch.nn.ModuleList()
        self.C_size = len(c_layers)
        for i in range(0, self.C_size-1):
            #adding dropout with prob dropbout_p

            self.C.append(torch.nn.Linear(in_features=input_shape,
                                          out_features=c_layers[i]))
            #self.C.append(torch.nn.Dropout(dropoutrate))
            input_shape = c_layers[i]

        self.last_C = torch.nn.Linear(in_features=input_shape,
                                     out_features=1)
        self.relu = torch.nn.LeakyReLU()
        self.sigm = torch.nn.Sigmoid()

    def forward_G(self, x):
        y = x
        for layer in self.G:
            y = self.relu(layer(y))
        inner_output = self.inner_output(y)
        return y, inner_output

    def forward_C(self, x):
        y = x
        for layer in self.C:
            y = self.relu(layer(y))
        return self.sigm(self.last_C(y))

    def forward(self, input1, input2):
        e1, inner_output1 = self.forward_G(input1)
        e2, inner_output2 = self.forward_G(input2)
        #absdiff = _torch_abs(e1, e2)
        # absdiff = _torch_abs2(e1 - e2)
        absdiff = torch.abs(e1-e2)
        r_t = self.forward_C(absdiff)
        return r_t, inner_output1, inner_output2


    def printgradnorm(self, grad_input, grad_output, fml):
        print('Inside ' + self.__class__.__name__ + ' backward')
        print('Inside class:' + self.__class__.__name__)
        print('')
        print('grad_input: ', type(grad_input))
        print('grad_input[0]: ', type(grad_input[0]))
        print('grad_output: ', type(grad_output))
        print('grad_output[0]: ', type(grad_output[0]))
        print('')
        print('grad_input size:', grad_input[0].size())
        print('grad_output size:', grad_output[0].size())
        print('grad_input norm:', grad_input[0].norm())
