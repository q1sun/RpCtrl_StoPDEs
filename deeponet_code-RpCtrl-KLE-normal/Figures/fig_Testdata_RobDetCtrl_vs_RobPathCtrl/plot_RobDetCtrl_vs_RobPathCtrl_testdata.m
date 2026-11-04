function plot_RobDetCtrl_vs_RobPathCtrl_testdata(plt_robdet_vs_robpath_testdata)

format short e

% testdata
load('Data/test_data_KLE_normal/Test_Data_KLE_Normal.mat')
% rob-det-ctrl for testdata
load('Checkpoints/RobDetCtrl_KLE_Normal.mat')
rob_det_ctrl_OptCtrl = rob_det_ctrl_OptCtrl;
% rob-path-ctrl for testdata
load('Figures/fig_Testdata_RobDetCtrl_vs_RobPathCtrl/Fig_Testdata_RobPathCtrl_KLE_Normal_Trained_Model.mat')
rob_path_ctrl_OptCtrl = rob_path_ctrl_OptCtrl_testdata;
% comparison results
load('Figures/fig_Testdata_RobDetCtrl_vs_RobPathCtrl/Fig_Testdata_RobDetCtrl_vs_RobPathCtrl_KLE_Normal.mat');  

% opengl('software')
% opengl('save','software')

if plt_robdet_vs_robpath_testdata == 1
    
    %---------------------------------------------------------------------------------------------------%
    % (c) statistical results of target match and control cost
    %---------------------------------------------------------------------------------------------------%
    figure('NumberTitle','off','Name','rob_det_ctrl v.s. path_ctrl using KLE coeff: target match and control cost');
    
    subplot(1,2,1)
    h1 = histogram(rob_det_ctrl_TargetMatch);
    hold on
    h2 = histogram(rob_path_ctrl_TargetMatch,20);
    
    h1.FaceColor = 	[8.5000e-01   3.2500e-01   9.8000e-02];
    h2.FaceColor = [0   4.4700e-01   7.4100e-01];
    h1.Normalization = 'probability';
    h2.Normalization = 'probability';
    set(gcf,'color','w')
    set(gca,'FontSize',18);
    % ax = axes;
    % ytickformat(ax, 'percentage');
    legend({'$\,\textnormal{RobDetCtrl}$','$\textnormal{RobPathCtrl}$'},'Interpreter','latex')
    ylabel({'$\textnormal{Percentage}$'},'Interpreter','latex')
    xlabel({'$\textnormal{Target match}$'},'Interpreter','latex')
    hold on
    
    subplot(1,2,2)
    h3 = histogram( rob_det_ctrl_CtrlCost); %,'Orientation','horizontal');
    h3.BinWidth = 0.1;
    hold on
    h4 = histogram(rob_path_ctrl_CtrlCost); %,'Orientation','horizontal');
    
    h3.FaceColor = 	[8.5000e-01   3.2500e-01   9.8000e-02];
    h4.FaceColor = [0   4.4700e-01   7.4100e-01];
    h3.Normalization = 'probability';
    h4.Normalization = 'probability';
    set(gcf,'color','w')
    set(gca,'FontSize',18);
    % ax = axes;
    % ytickformat(ax, 'percentage');
    legend({'$\,\textnormal{RobDetCtrl}$','$\textnormal{RobPathCtrl}$'},'Interpreter','latex')
    %legend({'$\ \| \hat{f}_{h,n}^*(x) \|_{L^2(D)}$','$\ \| f_{\omega,h}^*(x) \|_{L^2(D)}$'},'Interpreter','latex')
    ylabel({'\textnormal{Percentage}'},'Interpreter','latex')
    xlabel({'$\textnormal{Control cost}$'},'Interpreter','latex')
    %---------------------------------------------------------------------------------------------------%
    
    
    %---------------------------------------------------------------------------------------------------%
    % (f) comparison between RobDetCtrl and PathCtrl for (best and worst) coefficient realizations in test data
    %---------------------------------------------------------------------------------------------------%
    
    Lx = max(V_basis(1,:));
    Ly = max(V_basis(2,:));
    dx = 0:0.002:Lx;
    dy = 0:0.002:Ly;
    [qx,qy] = meshgrid(dx,dy);
    U = @(x,y) (sin(2*pi*x).*(cos(2*pi*y)-1)); % target function
        
    num = size(rob_det_ctrl_OptState,2);
    index_target_match = rob_det_ctrl_TargetMatch_Index_testdata(num-3:num);
    index_target_mismatch = rob_det_ctrl_TargetMatch_Index_testdata(1:4);
    
    fprintf('\n')
    fprintf(' Target Match for Testdata: coefficient index (worst case) for RobDetCtrl = %d',index_target_mismatch(1));
    fprintf('\n')
    fprintf(' Target Match for Testdata: coefficient index (best case) for RobDetCtrl = %d',index_target_match(4));
    fprintf('\n')
            
    figure('NumberTitle','off','Name','rob_det_ctrl v.s. rob_path_ctrl using KLE coeff: paper results');
    
    % coeff - worst case -  rob-det-ctrl
    subplot(4,4,1)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',test_data_inputs_Npts(:,index_target_mismatch(1)));
    qz = Ft(qx,qy);
    imagesc(qz)
    colormap jet
    set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
    set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
    colorbar off
    axis xy
    axis off
    axis equal
    set(gca,'FontSize',18);
    set(gcf,'color','w')
    hold on
    rectangle('position',[0 0 1 1] )
    ylabel({'$a(x,\omega)$'},'Interpreter','latex')
    %title({['state No.' num2str(index_target_mismatch(i))]});
    hold on
 
    % coeff - best case -  rob-det-ctrl
    subplot(4,4,2)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',test_data_inputs_Npts(:,index_target_match(4)));
    qz = Ft(qx,qy);
    imagesc(qz)
    colormap jet
    set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
    set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
    colorbar off
    axis xy
    axis off
    axis equal
    set(gca,'FontSize',18);
    set(gcf,'color','w')
    hold on
    rectangle('position',[0 0 1 1] )
    ylabel({'$a(x,\omega)$'},'Interpreter','latex')
    %title({['state No.' num2str(index_target_mismatch(i))]});
    hold on
    
    % target function
    subplot(4,4,4)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',U(V_basis(1,:),V_basis(2,:))');
    qz=Ft(qx,qy);
    imagesc(qz)
    colorbar off
    caxis([-2 2])
    colormap jet
    axis xy
    ax = gca;
    axis(ax,'off')
    axis equal
    set(gca,'FontSize',18);
    set(gcf,'color','w')
    rectangle('position',[0 0 1 1] )
    hold on
      
    % optimal rob-det-ctrl
    subplot(4,4,5)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',rob_det_ctrl_OptCtrl(:));
    qz=Ft(qx,qy);
    imagesc(qz)
    colorbar off
    colormap jet
    axis xy
    ax = gca;
    axis(ax,'off')
    set(get(ax,'XLabel'),'Visible','on')
    axis equal
    set(gca,'FontSize',18);
    set(gcf,'color','w')
    rectangle('position',[0 0 1 1] )

    % pathwise ctrl - worst case 
    subplot(4,4,7)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',rob_path_ctrl_OptCtrl(:,index_target_mismatch(1)));
    qz = Ft(qx,qy);
    imagesc(qz)
    colormap jet
    set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
    set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
    colorbar off
    axis xy
    axis off
    axis equal
    set(gca,'FontSize',18);
    set(gcf,'color','w')
    hold on
    rectangle('position',[0 0 1 1] )
    ylabel({'$a(x,\omega)$'},'Interpreter','latex')
    %title({['state No.' num2str(index_target_mismatch(i))]});
    hold on
 
    % pathwise ctrl - best case 
    subplot(4,4,8)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',rob_path_ctrl_OptCtrl(:,index_target_match(4)));
    qz = Ft(qx,qy);
    imagesc(qz)
    colormap jet
    set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
    set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
    colorbar off
    axis xy
    axis off
    axis equal
    set(gca,'FontSize',18);
    set(gcf,'color','w')
    hold on
    rectangle('position',[0 0 1 1] )
    ylabel({'$a(x,\omega)$'},'Interpreter','latex')
    %title({['state No.' num2str(index_target_mismatch(i))]});
    hold on

    % state - worst case -  rob-det-ctrl
    subplot(4,4,9)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',rob_det_ctrl_OptState(:,index_target_mismatch(1)));
    qz = Ft(qx,qy);
    imagesc(qz)
    colormap jet
    set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
    set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
    colorbar off
    axis xy
    axis off
    axis equal
    set(gca,'FontSize',18);
    set(gcf,'color','w')
    hold on
    %caxis([-2 2]) 
    rectangle('position',[0 0 1 1] )
    ylabel({'$a(x,\omega)$'},'Interpreter','latex')
    %title({['state No.' num2str(index_target_mismatch(i))]});
    hold on
 
    % state - best case -  rob-det-ctrl
    subplot(4,4,10)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',rob_det_ctrl_OptState(:,index_target_match(4)));
    qz = Ft(qx,qy);
    imagesc(qz)
    colormap jet
    set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
    set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
    colorbar off
    axis xy
    axis off
    axis equal
    set(gca,'FontSize',18);
    set(gcf,'color','w')
    hold on
    caxis([-2 2]) 
    rectangle('position',[0 0 1 1] )
    ylabel({'$a(x,\omega)$'},'Interpreter','latex')
    %title({['state No.' num2str(index_target_mismatch(i))]});
    hold on

    % state - worst case -  path-ctrl
    subplot(4,4,11)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',rob_path_ctrl_OptState(:,index_target_mismatch(1)));
    qz = Ft(qx,qy);
    imagesc(qz)
    colormap jet
    set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
    set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
    colorbar off
    axis xy
    axis off
    axis equal
    set(gca,'FontSize',18);
    set(gcf,'color','w')
    hold on
    caxis([-2 2]) 
    rectangle('position',[0 0 1 1] )
    ylabel({'$a(x,\omega)$'},'Interpreter','latex')
    %title({['state No.' num2str(index_target_mismatch(i))]});
    hold on
 
    % state - best case -  rob-det-ctrl
    subplot(4,4,12)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',rob_path_ctrl_OptState(:,index_target_match(4)));
    qz = Ft(qx,qy);
    imagesc(qz)
    colormap jet
    set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
    set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
    colorbar off
    axis xy
    axis off
    axis equal
    set(gca,'FontSize',18);
    set(gcf,'color','w')
    hold on
    caxis([-2 2]) 
    rectangle('position',[0 0 1 1] )
    ylabel({'$a(x,\omega)$'},'Interpreter','latex')
    %title({['state No.' num2str(index_target_mismatch(i))]});
    hold on
    
    
    % state err - worst case -  rob-det-ctrl
    subplot(4,4,13)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',rob_det_ctrl_OptState(:,index_target_mismatch(1)) - U(V_basis(1,:),V_basis(2,:))');
    qz = Ft(qx,qy);
    imagesc(qz)
    colormap jet
    set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
    set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
    colorbar off
    axis xy
    axis off
    axis equal
    set(gca,'FontSize',18);
    set(gcf,'color','w')
    hold on
    rectangle('position',[0 0 1 1] )
    ylabel({'$a(x,\omega)$'},'Interpreter','latex')
    %title({['state No.' num2str(index_target_mismatch(i))]});
    hold on
 
    % state err - best case -  rob-det-ctrl
    subplot(4,4,14)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',rob_det_ctrl_OptState(:,index_target_match(4)) - U(V_basis(1,:),V_basis(2,:))');
    qz = Ft(qx,qy);
    imagesc(qz)
    colormap jet
    set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
    set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
    colorbar off
    axis xy
    axis off
    axis equal
    set(gca,'FontSize',18);
    set(gcf,'color','w')
    hold on
    rectangle('position',[0 0 1 1] )
    ylabel({'$a(x,\omega)$'},'Interpreter','latex')
    %title({['state No.' num2str(index_target_mismatch(i))]});
    hold on

    % state err - worst case -  path-ctrl
    subplot(4,4,15)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',rob_path_ctrl_OptState(:,index_target_mismatch(1)) - U(V_basis(1,:),V_basis(2,:))');
    qz = Ft(qx,qy);
    imagesc(qz)
    colormap jet
    set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
    set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
    colorbar off
    axis xy
    axis off
    axis equal
    set(gca,'FontSize',18);
    set(gcf,'color','w')
    hold on
    rectangle('position',[0 0 1 1] )
    ylabel({'$a(x,\omega)$'},'Interpreter','latex')
    %title({['state No.' num2str(index_target_mismatch(i))]});
    hold on
 
    % state err - best case -  rob-det-ctrl
    subplot(4,4,16)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',rob_path_ctrl_OptState(:,index_target_match(4)) - U(V_basis(1,:),V_basis(2,:))');
    qz = Ft(qx,qy);
    imagesc(qz)
    colormap jet
    set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
    set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
    colorbar off
    axis xy
    axis off
    axis equal
    set(gca,'FontSize',18);
    set(gcf,'color','w')
    hold on
    rectangle('position',[0 0 1 1] )
    ylabel({'$a(x,\omega)$'},'Interpreter','latex')
    %title({['state No.' num2str(index_target_mismatch(i))]});
    hold on
    
end


%% auxiliray figures (activate if interested, not required for paper) 
% %---------------------------------------------------------------------------------------------------%
% % (d) worst-4 target match using rob_det_ctrl
% %---------------------------------------------------------------------------------------------------%
% figure('NumberTitle','off','Name','rob_det_ctrl v.s. rob_path_ctrl using KLE coeff: target mismatch');
%  
% for i = 1 : 4
%     
%     subplot(3,4,i)
%     Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',test_data_inputs_Npts(:,index_target_mismatch(i)));
%     qz = Ft(qx,qy);
%     imagesc(qz)
%     colormap jet
%     set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
%     set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
%     colorbar off
%     axis xy
%     axis off
%     axis equal
%     set(gca,'FontSize',18);
%     set(gcf,'color','w')
%     hold on
%     rectangle('position',[0 0 1 1] )
%     ylabel({'$a(x,\omega)$'},'Interpreter','latex')
%     %title({['state No.' num2str(index_target_mismatch(i))]});
%     hold on
%     
%     subplot(3,4,4+i)
%     Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',rob_det_ctrl_OptState(:,index_target_mismatch(i)));
%     qz = Ft(qx,qy);
%     imagesc(qz)
%     colormap jet
%     set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
%     set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
%     colorbar off
%     % caxis([-2 2]) % used for state only
%     axis xy
%     axis off
%     axis equal
%     set(gca,'FontSize',18);
%     set(gcf,'color','w')
%     hold on
%     rectangle('position',[0 0 1 1] )
%     ylabel({'$u^*_h(x,\omega)$'},'Interpreter','latex')
%     %title({['state No.' num2str(index_target_mismatch_rob_ctrl_KLE_coeff(i))]});
%     hold on
%     
%     subplot(3,4,8+i)
%     Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',rob_path_ctrl_OptState(:,index_target_mismatch(i)));
%     qz = Ft(qx,qy);
%     imagesc(qz)
%     colormap jet
%     set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
%     set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
%     colorbar off
%     caxis([-2 2]) % used for state only
%     axis xy
%     axis off
%     axis equal
%     set(gca,'FontSize',18);
%     set(gcf,'color','w')
%     hold on
%     rectangle('position',[0 0 1 1] )
%     ylabel({'$u^*_{\omega,h}(x)$'},'Interpreter','latex')
%     %title({['state No.' num2str(index_target_mismatch_rob_ctrl_KLE_coeff(i))]});
%     hold on
%     
% end
% %---------------------------------------------------------------------------------------------------%
%  
% %---------------------------------------------------------------------------------------------------%
% % (e) best-4 target match using rob_det_ctrl
% %---------------------------------------------------------------------------------------------------%
% figure('NumberTitle','off','Name','rob_det_ctrl v.s. rob_path_ctrl using KLE coeff: target match'); 
% 
% for i = 1 : 4
%     
%     subplot(3,4,i)
%     Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',test_data_inputs_Npts(:,index_target_match(i)));
%     qz = Ft(qx,qy);
%     imagesc(qz)
%     colormap jet
%     set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
%     set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
%     colorbar off
%     axis xy
%     axis off
%     axis equal
%     set(gca,'FontSize',18);
%     set(gcf,'color','w')
%     hold on
%     rectangle('position',[0 0 1 1] )
%     ylabel({'$a(x,\omega)$'},'Interpreter','latex')
%     %title({['state No.' num2str(index_target_mismatch(i))]});
%     hold on
%     
%     subplot(3,4,4+i)
%     Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',rob_det_ctrl_OptState(:,index_target_match(i)));
%     qz = Ft(qx,qy);
%     imagesc(qz)
%     colormap jet
%     set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
%     set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
%     colorbar off
%     caxis([-2 2]) % used for state only
%     axis xy
%     axis off
%     axis equal
%     set(gca,'FontSize',18);
%     set(gcf,'color','w')
%     hold on
%     rectangle('position',[0 0 1 1] )
%     ylabel({'$u^*_h(x,\omega)$'},'Interpreter','latex')
%     %title({['state No.' num2str(index_target_mismatch_rob_ctrl_KLE_coeff(i))]});
%     hold on
%     
%     subplot(3,4,8+i)
%     Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',rob_path_ctrl_OptState(:,index_target_match(i)));
%     qz = Ft(qx,qy);
%     imagesc(qz)
%     colormap jet
%     set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
%     set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
%     colorbar off
%     caxis([-2 2]) % used for state only
%     axis xy
%     axis off
%     axis equal
%     set(gca,'FontSize',18);
%     set(gcf,'color','w')
%     hold on
%     rectangle('position',[0 0 1 1] )
%     ylabel({'$u^*_{\omega,h}(x)$'},'Interpreter','latex')
%     %title({['state No.' num2str(index_target_mismatch_rob_ctrl_KLE_coeff(i))]});
%     hold on
%     
% end
% %---------------------------------------------------------------------------------------------------%


     

end




