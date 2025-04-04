import torch
import torch.nn as nn
from .layer import Net_RNA,Net_ATAC,Net_ATAC_MSE,Net_Protein,Net_omic_fusion_RNA_ATAC,Net_omic_fusion_RNA_ATAC_MSE,Net_omic_fusion_RNA_Protein
from .module import GAN_Discriminator
from tqdm import tqdm
import torch.nn.functional as F
from .utils import kl_loss,ZINBLoss,ZIPLoss
from .preprocess import data_preprocess

class MultiSP(nn.Module):
    def __init__(
            self,
            data,
            device= torch.device('cpu'),
            modality_type='RNA and Protein'
            ):
        super(MultiSP, self).__init__()
        self.data = data.copy()
        self.device = device
        self.pre_epoch=50
        self.epoch=50
        self.weight_decay=0.0000
        self.loss_weight =[1,1]
        self.modality_type=modality_type

        if self.modality_type=='RNA and Protein':
           self.k_neighbors=3
           self.learning_rate=0.0001
           self.d_learning_rate=0.0005
          
        if self.modality_type=='RNA and ATAC' or self.modality_type=='RNA and ATAC_P_mouse_brain':
           self.k_neighbors=6
           self.learning_rate=0.001
           self.d_learning_rate=0.005

        self.adata_omic1=self.data['adata_omics1']
        self.adata_omic2=self.data['adata_omics2']
        omic1_data_dict,omic2_data_dict=data_preprocess(self.adata_omic1,self.adata_omic2,self.modality_type,self.k_neighbors)

        self.adata_omic1_high_raw=omic1_data_dict['adata_omic_raw']
        self.feature1=torch.tensor(omic1_data_dict['feature'],dtype=torch.float32).to(self.device)
        self.feature1_raw=torch.tensor(omic1_data_dict['feature_raw'],dtype=torch.float32).to(self.device)
        self.scale_factor1=torch.tensor(omic1_data_dict['scale_factor'],dtype=torch.float32).to(self.device)
        self.adj_1=torch.tensor(omic1_data_dict['adj'],dtype=torch.float32).to(self.device)
        self.adj_feat1=torch.tensor(omic1_data_dict['adj_feat'],dtype=torch.float32).to(self.device)
        self.in_dim_1=[self.feature1_raw.shape[1],1024,128]

        if self.modality_type=='RNA and Protein':
           self.adata_omic2_raw=omic2_data_dict['adata_omic_raw']
           self.feature2_raw=torch.tensor(omic2_data_dict['feature_raw'],dtype=torch.float32).to(self.device)
           self.feature2=torch.tensor(omic2_data_dict['feature'],dtype=torch.float32).to(self.device)
           self.scale_factor2=torch.tensor(omic2_data_dict['scale_factor'],dtype=torch.float32).to(self.device)
           self.adj_2=torch.tensor(omic2_data_dict['adj'],dtype=torch.float32).to(self.device)
           self.adj_feat2=torch.tensor(omic2_data_dict['adj_feat'],dtype=torch.float32).to(self.device)
           self.in_dim_2=[self.feature2.shape[1],128]

        if self.modality_type=='RNA and ATAC':
           self.adata_omic2_raw=omic2_data_dict['adata_omic_raw']
           self.feature2=torch.tensor(omic2_data_dict['feature'],dtype=torch.float32).to(self.device)
           self.feature2_raw=torch.tensor(omic2_data_dict['feature_raw'],dtype=torch.float32).to(self.device)
           self.scale_factor2=torch.tensor(omic2_data_dict['scale_factor'],dtype=torch.float32).to(self.device)
           self.adj_2=torch.tensor(omic2_data_dict['adj'],dtype=torch.float32).to(self.device)
           self.adj_feat2=torch.tensor(omic2_data_dict['adj_feat'],dtype=torch.float32).to(self.device)
           self.in_dim_2=[self.feature2_raw.shape[1],1024,128]

        if self.modality_type=='RNA and ATAC_P_mouse_brain':
           self.adata_omic2_raw=omic2_data_dict['adata_omic_raw']
           self.feature2=torch.tensor(omic2_data_dict['feature'],dtype=torch.float32).to(self.device)
           self.adj_2=torch.tensor(omic2_data_dict['adj'],dtype=torch.float32).to(self.device)
           self.adj_feat2=torch.tensor(omic2_data_dict['adj_feat'],dtype=torch.float32).to(self.device)
           self.in_dim_2=[self.feature2.shape[1],128]
        
    def train(self):
        if self.modality_type=='RNA and Protein':
           self.model_omic1=Net_RNA(self.in_dim_1).to(self.device)
           self.model_omic2=Net_Protein(self.in_dim_2).to(self.device)
        if self.modality_type=='RNA and ATAC':
           self.model_omic1=Net_RNA(self.in_dim_1).to(self.device)
           self.model_omic2=Net_ATAC(self.in_dim_2).to(self.device)
        if self.modality_type=='RNA and ATAC_P_mouse_brain':
           self.model_omic1=Net_RNA(self.in_dim_1).to(self.device)
           self.model_omic2=Net_ATAC_MSE(self.in_dim_2).to(self.device)

        self.optimizer_omic1 = torch.optim.Adam(self.model_omic1.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)
        self.optimizer_omic2 = torch.optim.Adam(self.model_omic2.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)
        self.model_omic1.train()
        for epoch in tqdm(range(self.pre_epoch)):
            z1,mu1,logvar1,pi,disp,mean=self.model_omic1(self.feature1,self.adj_1,self.adj_feat1)
            kl_1=kl_loss(mu1,logvar1)
            recon_loss1=ZINBLoss(self.feature1_raw,mean,disp,pi,self.scale_factor1)
            loss1=recon_loss1+0.005*kl_1

            self.optimizer_omic1.zero_grad()
            loss1.backward()
            self.optimizer_omic1.step()

        self.model_omic2.train()
        if self.modality_type=='RNA and Protein':
           for epoch in tqdm(range(self.pre_epoch)):
               z2,mu2,logvar2,x_rec2=self.model_omic2(self.feature2,self.adj_2,self.adj_feat2)
               kl_2=kl_loss(mu2,logvar2)
               recon_loss2=F.mse_loss(self.feature2,x_rec2)
               loss2=recon_loss2+0.005*kl_2

               self.optimizer_omic2.zero_grad()
               loss2.backward()
               self.optimizer_omic2.step()
         

        if self.modality_type=='RNA and ATAC':
           for epoch in tqdm(range(self.pre_epoch)):
               z2,mu2,logvar2,pi,omega=self.model_omic2(self.feature2,self.adj_2,self.adj_feat2)
               kl_2=kl_loss(mu2,logvar2)
               recon_loss2=ZIPLoss(self.feature2_raw,pi,omega,self.scale_factor2)
               loss2=recon_loss2+0.005*kl_2

               self.optimizer_omic2.zero_grad()
               loss2.backward()
               self.optimizer_omic2.step()
        
        if self.modality_type=='RNA and ATAC_P_mouse_brain':
           for epoch in tqdm(range(self.pre_epoch)):
               z2,mu2,logvar2,x_rec2=self.model_omic2(self.feature2,self.adj_2,self.adj_feat2)
               kl_2=kl_loss(mu2,logvar2)
               recon_loss2=F.mse_loss(self.feature2,x_rec2)
               loss2=recon_loss2+0.005*kl_2

               self.optimizer_omic2.zero_grad()
               loss2.backward()
               self.optimizer_omic2.step()
      
       
        if self.modality_type=='RNA and ATAC':
               self.model=Net_omic_fusion_RNA_ATAC(self.in_dim_1,self.in_dim_2,self.model_omic1,self.model_omic2).to(self.device)
        if self.modality_type=='RNA and ATAC_P_mouse_brain':
               self.model=Net_omic_fusion_RNA_ATAC_MSE(self.in_dim_1,self.in_dim_2,self.model_omic1,self.model_omic2).to(self.device)
        if self.modality_type=='RNA and Protein':
               self.model=Net_omic_fusion_RNA_Protein(self.in_dim_1,self.in_dim_2,self.model_omic1,self.model_omic2).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)
        

        self.discriminator=GAN_Discriminator(self.in_dim_1[2]).to(self.device)
        self.optimizer_discriminator=torch.optim.RMSprop(self.discriminator.parameters(), lr=self.d_learning_rate)

        d_loss=torch.nn.CrossEntropyLoss(reduce='mean')

        self.model.train()
        if self.modality_type=='RNA and Protein':
           for epoch in tqdm(range(self.epoch)):
               #optimize the discriminator
               output,mu1,logvar1,mu2,logvar2=self.model(self.feature1,
                                                     self.feature2,self.adj_1,self.adj_2,self.adj_feat1,self.adj_feat2)
               z1=output['emb_latent_omics1']
               z2=output['emb_latent_omics2']
               label_1=torch.zeros(z1.size(0),dtype=torch.long).to(self.device)
               label_2=torch.ones(z2.size(0),dtype=torch.long).to(self.device)
               label=torch.cat((label_1,label_2),dim=0)
               pred=torch.cat((self.discriminator(z1),self.discriminator(z2)),dim=0)
               loss_d=d_loss(pred,label)
           
               self.optimizer_discriminator.zero_grad()
               loss_d.backward()
               self.optimizer_discriminator.step()
               
               #optimize the generator
               output,mu1,logvar1,mu2,logvar2=self.model(self.feature1,
                                                     self.feature2,self.adj_1,self.adj_2,self.adj_feat1,self.adj_feat2)
               loss1_rec=ZINBLoss(self.feature1_raw,output['emb_recon_mean'],output[ 'emb_recon_disp'],output[ 'emb_recon_pi'],self.scale_factor1)
               loss2_rec=F.mse_loss(self.feature2,output['emb_recon_omics2'])
               kl_1=kl_loss(mu1,logvar1)
               kl_2=kl_loss(mu2,logvar2)

               #caculate the adversarial loss
               z1=output['emb_latent_omics1']
               z2=output['emb_latent_omics2']
               pred=torch.cat((self.discriminator(z1),self.discriminator(z2)),dim=0)
               gloss_d=d_loss(pred,label)
             
               ad_loss=-gloss_d
               beta=1
               loss_recon=self.loss_weight[0]*(loss1_rec)+self.loss_weight[1]*(loss2_rec)+beta*(kl_1+kl_2)

               loss=loss_recon+0.01*ad_loss
               self.optimizer.zero_grad()
               loss.backward()
               self.optimizer.step()

        if self.modality_type=='RNA and ATAC':
           for epoch in tqdm(range(self.epoch)):
               #optimize the discriminator
               output,mu1,logvar1,mu2,logvar2=self.model(self.feature1,
                                                     self.feature2,self.adj_1,self.adj_2,self.adj_feat1,self.adj_feat2)
               z1=output['emb_latent_omics1']
               z2=output['emb_latent_omics2']
               label_1=torch.zeros(z1.size(0),dtype=torch.long).to(self.device)
               label_2=torch.ones(z2.size(0),dtype=torch.long).to(self.device)
               label=torch.cat((label_1,label_2),dim=0)
               pred=torch.cat((self.discriminator(z1),self.discriminator(z2)),dim=0)
               loss_d=d_loss(pred,label)
           
               self.optimizer_discriminator.zero_grad()
               loss_d.backward()
               self.optimizer_discriminator.step()
               
               #optimize the generator
               output,mu1,logvar1,mu2,logvar2=self.model(self.feature1,
                                                     self.feature2,self.adj_1,self.adj_2,self.adj_feat1,self.adj_feat2)
               
               loss1_rec=ZINBLoss(self.feature1_raw,output['emb_recon_mean'],output[ 'emb_recon_disp'],output[ 'emb_recon_pi'],self.scale_factor1)
               loss2_rec=ZIPLoss(self.feature2_raw,output['emb_recon_pi2'],output['emb_recon_omega'],self.scale_factor2)

               kl_1=kl_loss(mu1,logvar1)
               kl_2=kl_loss(mu2,logvar2)
               
               #caculate the adversarial loss
               z1=output['emb_latent_omics1']
               z2=output['emb_latent_omics2']
               pred=torch.cat((self.discriminator(z1),self.discriminator(z2)),dim=0)
               gloss_d=d_loss(pred,label)
               ad_loss=-gloss_d
             
               beta=1
               loss_recon=self.loss_weight[0]*(loss1_rec)+self.loss_weight[1]*(loss2_rec)+beta*(kl_1+kl_2)

               loss=loss_recon+0.01*ad_loss
               self.optimizer.zero_grad()
               loss.backward()
               self.optimizer.step()

        if self.modality_type=='RNA and ATAC_P_mouse_brain':
           for epoch in tqdm(range(self.epoch)):
               #optimize the discriminator
               output,mu1,logvar1,mu2,logvar2=self.model(self.feature1,
                                                     self.feature2,self.adj_1,self.adj_2,self.adj_feat1,self.adj_feat2)
               z1=output['emb_latent_omics1']
               z2=output['emb_latent_omics2']
               label_1=torch.zeros(z1.size(0),dtype=torch.long).to(self.device)
               label_2=torch.ones(z2.size(0),dtype=torch.long).to(self.device)
               label=torch.cat((label_1,label_2),dim=0)
               pred=torch.cat((self.discriminator(z1),self.discriminator(z2)),dim=0)
               loss_d=d_loss(pred,label)
           
               self.optimizer_discriminator.zero_grad()
               loss_d.backward()
               self.optimizer_discriminator.step()
             
               
               #optimize the generator
               output,mu1,logvar1,mu2,logvar2=self.model(self.feature1,
                                                     self.feature2,self.adj_1,self.adj_2,self.adj_feat1,self.adj_feat2)
               
               loss1_rec=ZINBLoss(self.feature1_raw,output['emb_recon_mean'],output[ 'emb_recon_disp'],output[ 'emb_recon_pi'],self.scale_factor1)
               loss2_rec=F.mse_loss(self.feature2,output['emb_recon_omics2'])

               kl_1=kl_loss(mu1,logvar1)
               kl_2=kl_loss(mu2,logvar2)
               
               #caculate the adversarial loss
               z1=output['emb_latent_omics1']
               z2=output['emb_latent_omics2']
               pred=torch.cat((self.discriminator(z1),self.discriminator(z2)),dim=0)
               gloss_d=d_loss(pred,label)
               ad_loss=-gloss_d
   
               beta=1
               loss_recon=self.loss_weight[0]*(loss1_rec)+self.loss_weight[1]*(loss2_rec)+beta*(kl_1+kl_2)

               loss=loss_recon+0.01*ad_loss
               self.optimizer.zero_grad()
               loss.backward()
               self.optimizer.step()

               
        
        self.model.eval()
        with torch.no_grad():
             output,mu1,logvar1,mu2,logvar2=self.model(self.feature1,self.feature2,
                                                 self.adj_1,self.adj_2,self.adj_feat1,self.adj_feat2)  
             z1=F.normalize(output['emb_latent_omics1'], p=2, eps=1e-12, dim=1)
             z2=F.normalize(output['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
             z=F.normalize(output['emb_latent_fusion'], p=2, eps=1e-12, dim=1)
            

        self.adata_omic1_high_raw.obsm['MultiSP_latent']=z.cpu().numpy()
        self.adata_omic1_high_raw.obsm['denoised_expr']=output['emb_recon_mean'].cpu().numpy()

       
        return self.adata_omic1_high_raw,self.adata_omic2_raw
    