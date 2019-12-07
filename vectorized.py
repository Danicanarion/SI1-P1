x = p_X_training[i,:][np.newaxis, :].astype(float)
                # Forward
                out = [(x, x)]
                for layer in self.hidden_layers_:
                    print(out[-1][1])
                    z = layer._net_input(out[-1][1])
                    a = layer._activation(z)
                    out.append((z, a))

                z = self.output_layer_._net_input(out[-1][1])
                a = self.output_layer_._activation(z)

                out.append((z, a))

                # Backward
                delta = []
                a = out[-1][1]
               # print(p_Y_training[i], a, z, sigDx(z))
                delta.insert(0, l2_cost[1](
                    p_Y_training[i], a) * sigDx(z))
                _w = self.output_layer_.w[1:,:]
               # print(delta[0])

                #print(p_Y_training[i], a)
                # Gradient descent
                self.output_layer_.w[0,:] = self.output_layer_.w[0,:] - np.mean(delta[0]) * self.eta
                self.output_layer_.w[1:,:] = self.output_layer_.w[1:,:] + self.eta * out[-2][1].T @ delta[0]
                
                #print(delta[0])

                for layer in reversed(range(p_number_hidden_layers)):
                    #print(layer)
                    z = out[layer + 1][0]
                    a = out[layer + 1][1]

                    print(delta[0], _w.T, delta[0] @ _w.T)
                    delta.insert(0, delta[0] @ _w.T * sigDx(z))
                    sys.exit(1)
                    _w = self.hidden_layers_[layer].w[1:,:]
                    

                    self.hidden_layers_[layer].w[0,:] = self.hidden_layers_[layer].w[0,:] - np.mean(delta[0]) * self.eta
                    self.hidden_layers_[layer].w[1:,:] = self.hidden_layers_[
                            layer].w[1:,:] + self.eta * out[layer][1].T @ delta[0]
            

