import torch

class ESNNModel(torch.nn.Module):
    def __init__(self, X, Y, networklayers=[13, 13]):
        """

        """
        super(ESNNModel, self).__init__()
        input_shape = X.shape[1]
        g_layers = networklayers
        c_layers = networklayers
        if isinstance(networklayers[0], list):
            # this means user has specified different layers
            # for g and c..
            g_layers = networklayers[0]
            c_layers = networklayers[1]


        # input1 = Input(shape=(X.shape[1],), dtype="float32")
        # input2 = Input(shape=(X.shape[1],), dtype="float32")

        # given S(x,y) = C(G(x),G(y)):
        # \hat{x} = G(x)
        self.G = torch.nn.ModuleList()
        for networklayer in g_layers:
            self.G.append(torch.nn.Linear(in_features=input_shape,
                                          out_features=networklayer))
            input_shape = networklayer

        self.inner_output = torch.nn.Linear(in_features=input_shape,
                                            out_features=Y.shape[1])

        self.C = torch.nn.ModuleList()
        for networklayer in c_layers:
            self.C.append(torch.nn.Linear(in_features=input_shape,
                                          out_features=networklayer))
            input_shape = networklayer

        self.C.append(torch.nn.Linear(in_features=input_shape,
                                      out_features=1))
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()

    def forward_G(self, x):
        y = x
        for i in range(len(self.G)):
            if torch.any(torch.isnan(y)):
                print("isnan")
            y = self.relu(self.G[i](y))
        inner_output = torch.softmax(self.inner_output(y), dim=1)
        return y, inner_output

    def forward_C(self, x):
        y = x
        for i in range(len(self.C)-1):
            y = self.relu(self.C[i](y))
        return self.sigm(self.C[len(self.C)-1](y))

    def forward(self, input1, input2):
        e1, inner_output1 = self.forward_G(input1)
        if torch.any(torch.isinf(torch.log(inner_output1))):
            print("heh")
        e2, inner_output2 = self.forward_G(input2)
        #absdiff = _torch_abs(e1, e2)
        # absdiff = _torch_abs2(e1 - e2)
        absdiff = torch.abs(e1-e2)
        r_t = self.forward_C(absdiff)
        return r_t, inner_output1, inner_output2
