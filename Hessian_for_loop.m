%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                         %  Neural Network  %
                         %   Coursework 2   %
                         %  Maxime Carriere %
                         %    33 56 70 21   %
                    
                    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



clear all, close all, clc

%% LOAD AND NORMALIZE THE DATA

load sunspot.dat
year=sunspot(:,1); relNums=sunspot(:,2); %plot(year,relNums)
ynrmv=mean(relNums(:)); sigy=std(relNums(:)); 
nrmY=relNums; %nrmY=(relNums(:)-ynrmv)./sigy; 
ymin=min(nrmY(:)); ymax=max(nrmY(:)); 
relNums=2.0*((nrmY-ymin)/(ymax-ymin)-0.5);


%% INITIALIZED INPUT AND DESIRED

Ss   = relNums';
Input_dim = 10; % input dimension
out_dim = length(Ss)-Input_dim; % output dimension
y    = zeros(1,278);

for i=1:out_dim
   y(i)=Ss(i+Input_dim);
   for j=1:Input_dim
       x(i,j) = Ss(i-j+Input_dim); 
   end
end 


Inputs1=[ones(1,size(x,1)); x']'; 
Desired= y';
[Nb_Patterns, Nb_Inputs1]= size(Inputs1); 
Nb_Outputs= size(Desired,2);



%% INITIALIZED HIDDEN NODES AND WEIGHTS

Nb_Hidden = 5;
Weights_1 =  0.5*(rand(Nb_Hidden, Nb_Inputs1)-0.5); % from input to hidden nodes
Weights_2 =  0.5*(rand(Nb_Outputs, Nb_Inputs1)-0.5); % from hidden to final node


for e = 1:200 % epochs
    
    for p = 1 : Nb_Patterns % 278 
        
        
        %% FEED-FORWARD PROPAGATION
        
        % INPUTS TO HIDDEN NODES
        
            for h = 1:Nb_Hidden % 5 
                
                for i = 1: Nb_Inputs1 % 11
        
                         SOut_1(h,i)  =   Weights_1(h,i)*Inputs1(p,i);       
                end
                
                 SOut(1,h) = 1/(1 + exp(-(sum(SOut_1(h,:)))));
            end
            
            
        Inputs2 = [ones(1) SOut];
        
        

       
        % HIDDEN TO FINAL NODE
        
        
        
         for h = 1:Nb_Hidden +1 % 6 
        
                 Out_1 (h) = Weights_2(1,h) * Inputs2(1,h);
                 
         end
        
                 Out (p,:) = sum(Out_1)';
                 
                 
                 
          % INITIALIZED PARAMETERS
          
          LR = 1; 
          Beta = 1; 
          Error(p,:)  = Desired(p,:) - Out(p,:); 
         
            
          
          
          %% BACKWARD PROPAGATION
          
          
          for h = 1:Nb_Hidden +1 % 6 
              
              Beta_Nodes (h) = (1-Inputs2(h).^2).*(Weights_2(h)*Beta);
          
          end
        

          %% DELTA WEIGHT 
          
          % WEIGHTS_1 : INPUT TO HIDDEN NODES
          
          
          for h = 1:Nb_Hidden % 5 
                
                for i = 1: Nb_Inputs1 % 11
                    
                    Delta_Weights_1(h,i) = Beta_Nodes(h+1) * Inputs1(p,i);
                    
                end
                
          end
          
          
          % WEIGHTS_2 : HIDDEN NODES TO FINAL NODE
          
          
          for h = 1:Nb_Hidden +1 % 6
              
              Delta_Weights_2 (h) = Beta * Inputs2(h);
              
          end
          
            
          
          %% JACOBIAN
          
          % Matrix Delta Weights
          
           
           Delta_Weights_1_2 = Delta_Weights_1'; 
           Vector_Delta_Weights_1 = Delta_Weights_1_2(:);
           Matrix_Delta_Weights_1 (p,:) = Vector_Delta_Weights_1';
           Matrix_Delta_Weights_2 (p,:) = Delta_Weights_2(:,:);

        
    end
    
           % Jacobian 
           
           Jacobian = [Matrix_Delta_Weights_2 Matrix_Delta_Weights_1];




            %% HESSIAN MATRIX
            
            
            Hessian = Jacobian'*Jacobian ; 
            
            disp(Hessian(1:3,1:3))
            
            Inv_Hessian = (inv(Hessian +(0.001*eye(size(Hessian))))/Nb_Patterns)/100;
            
            
            
    
            
    %% SECOND PART
    
            
    for p = 1 : Nb_Patterns % 278 
        
        
        %% FEED-FORWARD PROPAGATION 2
        
        % INPUTS TO HIDDEN NODES
        
            for h = 1:Nb_Hidden % 5 
                
                for i = 1: Nb_Inputs1 % 11
        
                         SOut_1_2(h,i)  =   Weights_1(h,i)*Inputs1(p,i);          
                         %SOut_2(1,h) = 1-2/(exp(2*SOut_1_2(h,i)));
                          
                end
                
                 SOut_2(1,h) = 1/(1 + exp(-(sum(SOut_1_2(h,:)))));
                 
            end
            
             
        Inputs2 = [ones(1) SOut_2];
        Inputs2balanl (p,:) = [ones(1) SOut_2];
            
          
        % HIDDEN TO FINAL NODE
        
        
        
         for h = 1:Nb_Hidden +1 % 6 
        
                 Out_1_2 (h) = Weights_2(1,h) * Inputs2(1,h);
                 
         end

                 Out_3 = sum(Out_1_2);
                 Out_2 (p,:) = Out_3;
                     
                          
                 
                 
          % ERROR
        
          Error_1(p,:)  = Desired(p,:) - Out_2(p,:); 
          Beta = Error_1; 
          TSS (e) = sum(sum( Error.^2 ));
          TSS_1 (e) = TSS(e); 
          
          
          %% BACKWARD PROPAGATION
          
          
          for h = 1:Nb_Hidden +1 % 6 
              
              Beta_Nodes_2 (h) = (1-Inputs2(h).^2).*(Weights_2(h)*Beta(p,:));
          
          end
        
          
          
          


          %% DELTA WEIGHTS 
          % ERROR ==> WEIGHTS_1 : INPUT TO HIDDEN NODES
          
          
          for h = 1:Nb_Hidden % 5 
                
                for i = 1: Nb_Inputs1 % 11
                    
                    Delta_Weights_1(h,i) = Beta_Nodes_2(h+1) * Inputs1(p,i);
                    
                end
                
          end
          
          
          % ERROR ==> WEIGHTS_2 : HIDDEN NODES TO FINAL NODE
          
          
          for h = 1:Nb_Hidden +1 % 6
              
              Delta_Weights_2 (h) = Beta(p,:) * Inputs2(h);
              
          end
          
          

    
          %% UPDATE OF WEIGHTS 
          
          % WEIGHTS_1 : INPUT TO HIDDEN NODES
          
          
          for     i= 1: Nb_Hidden % 5
              
                for     j= 1: Nb_Inputs1 % 11
    
        Weights_1(i,j)= Inv_Hessian(Nb_Inputs1*i+j-Nb_Hidden ,Nb_Inputs1*i+j-Nb_Hidden) * Delta_Weights_1(i,j) + Weights_1(i,j);
        
                end
        
            end
          
          
          
          % WEIGHTS_2 : INPUT TO HIDDEN NODES
          
         
          for h = 2:Nb_Hidden+1 % 6
          
          Weights_2 (h) = Weights_2(h) + Inv_Hessian(h,h)*Delta_Weights_2(h);
          
          end
          
          
          
          
end             
  
fprintf('Epoch %3d:  Error = %f\n',e,TSS);



end 



plot(year(11:288),Desired,year(11:288),Out)
title('Sunspot Data')


figure (2)
plot(1:e,TSS_1)
hold on
