import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .module import GraphConvolution



class AttentionLayer(torch.nn.Module):
    
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)
        
    def forward(self, emb1, emb2):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        self.emb = torch.cat(emb, dim=1)
        
        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu=  torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)  
        
        emb_combined = torch.matmul(torch.transpose(self.emb,1,2), torch.unsqueeze(self.alpha, -1))
    
        return torch.squeeze(emb_combined), self.alpha 
            
    

class vae_Encoder(torch.nn.Module):
    def __init__(self,int_dim,out_dim):
        super(vae_Encoder,self).__init__()
        self.enc_mu=nn.Linear(int_dim, out_dim)
        self.enc_logvar=nn.Linear(int_dim, out_dim)
    
    def forward(self, x):
        mu=self.enc_mu(x)
        logvar=self.enc_logvar(x)
        return mu,logvar
    
class vae_Decoder(torch.nn.Module):
    def __init__(self,int_dim,out_dim):
        super(vae_Decoder,self).__init__()
        self.lin=nn.Linear(int_dim, out_dim)
    def forward(self, x):
        z=self.lin(x)
        return z



class ZINBDecoder(torch.nn.Module):
    def __init__(self, d_in,d_hid,d_out):
        super(ZINBDecoder, self).__init__()
        self.lin1=nn.Sequential(nn.Linear(d_in, d_hid),
                                 nn.BatchNorm1d(d_hid),
                                nn.ELU(inplace=True),
                                nn.Dropout(p=0.0)
                                )
        
        self.pi=nn.Linear(d_hid,d_out)
        self.disp=nn.Linear(d_hid,d_out)
        self.mean=nn.Linear(d_hid,d_out)
        
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self,x):
        x=self.lin1(x)
        pi= torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
       
        return [pi, disp, mean]

class ZIPDecoder(torch.nn.Module):
    def __init__(self, d_in,d_hid,d_out):
        super(ZIPDecoder, self).__init__()
        self.lin=nn.Sequential(nn.Linear(d_in, d_hid),
                                nn.BatchNorm1d(d_hid),
                                nn.ELU(inplace=True),
                                nn.Dropout(p=0.0)
                                )
        self.pi=nn.Linear(d_hid,d_out)
        self.rho_output=nn.Linear(d_hid,d_out)
        self.peak_bias = nn.Parameter(torch.randn(1, d_out))
        nn.init.xavier_normal_(self.peak_bias)
        

    def forward(self,x):
        x=self.lin(x)
        pi= torch.sigmoid(self.pi(x))
        rho = self.rho_output(x)
        omega = F.softmax(rho+ self.peak_bias, dim=-1)
        return [pi,omega]


    
class Encoder(torch.nn.Module):
    def __init__(self,d_in,d_out):
        super(Encoder, self).__init__()
        self.conv=GraphConvolution(d_in,d_out)
        self.attention=AttentionLayer(d_out,d_out)
      
        
    def forward(self, x, adj,adj_feat):
        x1=self.conv(x,adj)
        x2=self.conv(x,adj_feat)
        x,_=self.attention(x1,x2)
        return x

class Decoder(torch.nn.Module):
    def __init__(self,d_in,d_out):
        super(Decoder, self).__init__()
        self.lin=nn.Sequential(nn.Linear(d_in, d_out)
                                )
      
    def forward(self,x):
        x=self.lin(x)
        return x
    



class Net_RNA(torch.nn.Module):
    def __init__(self,in_dims):
        super(Net_RNA,self).__init__()
        [dim_1,dim_2,dim_3] = in_dims
        self.encoder=Encoder(100,dim_3)
        self.vaencoder=vae_Encoder(dim_3,dim_3)
        self.decoder=ZINBDecoder(dim_3,dim_2,dim_1)

  
    def forward(self,feat,adj,adj_feat):
        z= self.encoder(feat,adj,adj_feat)
        z=F.elu(z)
        z=F.dropout(z,p=0.0,training=self.training)
        mu,logvar=self.vaencoder(z)
        z=self.reparametrize(mu,logvar)
        pi,disp,mean=self.decoder(z)
        return z,mu,logvar,pi,disp,mean

    def reparametrize(self, mu, sigma):
        sigma = torch.exp(sigma/2)
        eps = torch.randn_like(sigma)
        if self.training:
           z = mu + eps * sigma
        else:
           z=mu
        return z

class Net_ATAC(torch.nn.Module):
    def __init__(self,in_dims):
        super(Net_ATAC,self).__init__()
        [dim_1,dim_2,dim_3] = in_dims
        self.encoder=Encoder(100,dim_3)
        self.vaencoder=vae_Encoder(dim_3,dim_3)
        self.decoder=ZIPDecoder(dim_3,dim_2,dim_1)
      
  
    def forward(self,feat,adj,adj_feat):
        z= self.encoder(feat,adj,adj_feat)
        z=F.elu(z)
        z=F.dropout(z,p=0.0,training=self.training)
        mu,logvar=self.vaencoder(z)
        z=self.reparametrize(mu,logvar)
        pi,omega=self.decoder(z)
          
        return z,mu,logvar,pi,omega
    
    def reparametrize(self, mu, sigma):
        sigma = torch.exp(sigma/2)
        eps = torch.randn_like(sigma)
        if self.training:
           z = mu + eps * sigma
        else:
           z=mu
        return z

class Net_ATAC_MSE(torch.nn.Module):
    def __init__(self,in_dims):
        super(Net_ATAC_MSE,self).__init__()
        [dim_1,dim_2] = in_dims
        self.encoder=Encoder(50,dim_2)
        self.vaencoder=vae_Encoder(dim_2,dim_2)
        self.decoder=Decoder(dim_2,dim_1)
      
  
    def forward(self,feat,adj,adj_feat):
        z= self.encoder(feat,adj,adj_feat)
        z=F.elu(z)
        z=F.dropout(z,p=0.0,training=self.training)
        mu,logvar=self.vaencoder(z)
        z=self.reparametrize(mu,logvar)
        x_rec=self.decoder(z)
          
        return z,mu,logvar,x_rec
    
    def reparametrize(self, mu, sigma):
        sigma = torch.exp(sigma/2)
        eps = torch.randn_like(sigma)
        if self.training:
           z = mu + eps * sigma
        else:
           z=mu
        return z

class Net_Protein(torch.nn.Module):
    def __init__(self,in_dims):
        super(Net_Protein,self).__init__()
        [dim_1,dim_2] = in_dims
        self.encoder=Encoder(dim_1,dim_2)
        self.vaencoder=vae_Encoder(dim_2,dim_2)
        self.decoder=Decoder(dim_2,dim_1)
    def forward(self,feat,adj,adj_feat):
        z=self.encoder(feat,adj,adj_feat)
        z=F.elu(z)
        z=F.dropout(z,p=0.0,training=self.training)
        mu,logvar=self.vaencoder(z)
        z=self.reparametrize(mu,logvar)
        x_rec=self.decoder(z)
       
        return z,mu,logvar,x_rec
    def reparametrize(self, mu, sigma):
        sigma = torch.exp(sigma/2)
        eps = torch.randn_like(sigma)
        if self.training:
           z = mu + eps * sigma
        else:
           z=mu
        return z


class Net_omic_fusion_RNA_ATAC(torch.nn.Module):
    def __init__(self,omic1_in_dims,omic2_in_dims,pre_model_1,pre_model_2):
        super(Net_omic_fusion_RNA_ATAC,self).__init__()
        self.omic1= Net_RNA(omic1_in_dims)
        self.omic2= Net_ATAC(omic2_in_dims)
        self.fusion=nn.Linear(omic1_in_dims[2]+omic2_in_dims[2],omic1_in_dims[2])
        self.load_pretrained(pre_model_1,pre_model_2)

    
    def load_pretrained(self,pre_model_1,pre_model_2):
        self.omic1.load_state_dict(pre_model_1.state_dict())
        self.omic2.load_state_dict(pre_model_2.state_dict())

       
    def forward(self,feat1,feat2,adj_1,adj_2,adj_feat1,adj_feat2):
        z1,mu1,logvar1,pi1,disp1,mean1=self.omic1(feat1,adj_1,adj_feat1)
        z2,mu2,logvar2,pi2,omega=self.omic2(feat2,adj_2,adj_feat2)
        z=self.fusion(torch.cat((z1,z2),dim=1))
        pi1,disp1,mean1=self.omic1.decoder(z)
        pi2,omega=self.omic2.decoder(z)
        results = {'emb_latent_omics1':z1,
                       'emb_latent_omics2':z2,
                       'emb_latent_fusion':z,
                       'emb_recon_pi': pi1,
                       'emb_recon_disp':disp1,
                       'emb_recon_mean':mean1,
                       'emb_recon_pi2':pi2,
                       'emb_recon_omega':omega
                      }
        return results,mu1,logvar1,mu2,logvar2
       
class Net_omic_fusion_RNA_ATAC_MSE(torch.nn.Module):
    def __init__(self,omic1_in_dims,omic2_in_dims,pre_model_1,pre_model_2):
        super(Net_omic_fusion_RNA_ATAC_MSE,self).__init__()
        self.omic1= Net_RNA(omic1_in_dims)
        self.omic2= Net_ATAC_MSE(omic2_in_dims)
        self.fusion=nn.Linear(omic1_in_dims[2]+omic2_in_dims[1],omic1_in_dims[2])
        self.load_pretrained(pre_model_1,pre_model_2)

    
    def load_pretrained(self,pre_model_1,pre_model_2):
        self.omic1.load_state_dict(pre_model_1.state_dict())
        self.omic2.load_state_dict(pre_model_2.state_dict())

       
    def forward(self,feat1,feat2,adj_1,adj_2,adj_feat1,adj_feat2):
        z1,mu1,logvar1,pi1,disp1,mean1=self.omic1(feat1,adj_1,adj_feat1)
        z2,mu2,logvar2,x_rec2=self.omic2(feat2,adj_2,adj_feat2)
        z=self.fusion(torch.cat((z1,z2),dim=1))
        pi1,disp1,mean1=self.omic1.decoder(z)
        x_rec2=self.omic2.decoder(z)
        results = {'emb_latent_omics1':z1,
                       'emb_latent_omics2':z2,
                       'emb_latent_fusion':z,
                       'emb_recon_pi': pi1,
                       'emb_recon_disp':disp1,
                       'emb_recon_mean':mean1,
                       'emb_recon_omics2':x_rec2
                       }
        return results,mu1,logvar1,mu2,logvar2
       

class Net_omic_fusion_RNA_Protein(torch.nn.Module):
    def __init__(self,omic1_in_dims,omic2_in_dims,pre_model_1,pre_model_2):
        super(Net_omic_fusion_RNA_Protein,self).__init__()
        self.omic1= Net_RNA(omic1_in_dims)
        self.omic2= Net_Protein(omic2_in_dims)
        self.fusion=nn.Linear(omic1_in_dims[2]+omic2_in_dims[1],omic1_in_dims[2])
        self.load_pretrained(pre_model_1,pre_model_2)

    
    def load_pretrained(self,pre_model_1,pre_model_2):
        self.omic1.load_state_dict(pre_model_1.state_dict())
        self.omic2.load_state_dict(pre_model_2.state_dict())

       
    def forward(self,feat1,feat2,adj_1,adj_2,adj_feat1,adj_feat2):
          z1,mu1,logvar1,pi1,disp1,mean1=self.omic1(feat1,adj_1,adj_feat1)
          z2,mu2,logvar2,x_rec2=self.omic2(feat2,adj_2,adj_feat2)
          z=self.fusion(torch.cat((z1,z2),dim=1))
          pi1,disp1,mean1=self.omic1.decoder(z)
          x_rec2=self.omic2.decoder(z)
          results = {'emb_latent_omics1':z1,
                       'emb_latent_omics2':z2,
                       'emb_latent_fusion':z,
                       'emb_recon_pi': pi1,
                       'emb_recon_disp':disp1,
                       'emb_recon_mean':mean1,
                       'emb_recon_omics2':x_rec2
                       }
          return results,mu1,logvar1,mu2,logvar2





