# class HistoryPlacePolicyValue(nn.Module):

#     def __init__(self, cvae_config, fetching_gpt_config, value_config):
#         super().__init__()   
#         self.cvae = CVAE(cvae_config)
#         self.fetching_gpt = GPT2FetchingConditioner(fetching_gpt_config)
#         self.value_net = ValueNet(value_config)


#     def forward(self, x, *args, **kargs):
#         """forward
#         (For training)

#         To forward the value only, do the following:
#         ```
#         c = model.fetching_gpt(*args, **kargs)
#         value = model.value_net(c)
#         ```


#         Args:
#             x (th.Tensor): Label
#             *args, **kargs (th.Tensor): Input to the GPT.

#         Returns:
#             recon_x (th.Tensor): Reconstructed `x`
#             mean (th.Tensor): mean
#             log_var (th.Tensor): log var
#             value (th.Tensor): Predicted value
#         """
        
#         c = self.fetching_gpt(*args, **kargs)
#         recon_x, mean, log_var = self.cvae(x, c)
#         value = self.value_net(c)

#         return recon_x, mean, log_var, value
    

#     def inference(self, *args, **kargs):
#         """For inference
        
#         Args:
#             *args, **kargs (th.Tensor): Input to the GPT

#         Returns:
#             recon_x (th.Tensor): Reconstructed `x`
#             value (th.Tensor): Predicted value
#         """
#         c = self.fetching_gpt(*args, **kargs)
#         recon_x = self.cvae.inference(c)
#         value = self.value_net(c)

#         return recon_x, value

    
#     def inference_value_only(self, *args, **kargs):
#         """For inference of value only.

#         Args:
#             *args, **kargs (th.Tensor): Input to the GPT

#         Returns:
#             value (th.Tensor): Predicted value
#         """
#         c = self.fetching_gpt(*args, **kargs)
#         value = self.value_net(c)

#         return value