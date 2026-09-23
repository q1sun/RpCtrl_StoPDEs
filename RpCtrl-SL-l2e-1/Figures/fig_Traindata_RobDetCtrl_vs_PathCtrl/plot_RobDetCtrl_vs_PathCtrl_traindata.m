function plot_RobDetCtrl_vs_PathCtrl_traindata(plt_rob_vs_path_traindata)

format short e

load('Figures/fig_Traindata_RobDetCtrl_vs_PathCtrl/Fig_Traindata_RobDetCtrl_KLE_Normal.mat');
load('Figures/fig_Traindata_RobDetCtrl_vs_PathCtrl/Fig_Traindata_PathCtrl_KLE_Normal.mat');
load('Figures/fig_Traindata_RobDetCtrl_vs_PathCtrl/Fig_Comparison_RobDetCtrl_vs_PathCtrl_KLE_Normal.mat');

Lx = max(V_basis(1,:));
Ly = max(V_basis(2,:));
dx = 0:0.002:Lx;
dy = 0:0.002:Ly;
[qx,qy] = meshgrid(dx,dy);
U = @(x,y) (sin(2*pi*x).*(cos(2*pi*y)-1)); % target function

% opengl('software')
% opengl('save','software')

if plt_rob_vs_path_traindata == 1
    
    %---------------------------------------------------------------------------------------------------%
    % (c) statistical results of target match and control cost
    %---------------------------------------------------------------------------------------------------%
    figure('NumberTitle','off','Name','rob_det_ctrl v.s. path_ctrl using KLE-normal: target match and control cost');
    
    subplot(1,2,1)
    h1 = histogram(rob_det_ctrl_TargetMatch);
    hold on
    h2 = histogram(path_ctrl_TargetMatch,20);
    
    h1.FaceColor = 	[8.5000e-01   3.2500e-01   9.8000e-02];
    h2.FaceColor = [0   4.4700e-01   7.4100e-01];
    h1.Normalization = 'probability';
    h2.Normalization = 'probability';
    set(gcf,'color','w')
    set(gca,'FontSize',18);
    % ax = axes;
    % ytickformat(ax, 'percentage');
    legend({'$\ \textnormal{RobDetCtrl}$','$\ \ \ \textnormal{PathCtrl}$'},'Interpreter','latex')
    ylabel({'$\textnormal{Percentage}$'},'Interpreter','latex')
    xlabel({'$\textnormal{Target match}$'},'Interpreter','latex')
    hold on
    
    subplot(1,2,2)
    h3 = histogram( rob_det_ctrl_CtrlCost); %,'Orientation','horizontal');
    h3.BinWidth = 0.1;
    hold on
    h4 = histogram(path_ctrl_CtrlCost); %,'Orientation','horizontal');
    
    h3.FaceColor = 	[8.5000e-01   3.2500e-01   9.8000e-02];
    h4.FaceColor = [0   4.4700e-01   7.4100e-01];
    h3.Normalization = 'probability';
    h4.Normalization = 'probability';
    set(gcf,'color','w')
    set(gca,'FontSize',18);
    % ax = axes;
    % ytickformat(ax, 'percentage');
    legend({'$\ \textnormal{RobDetCtrl}$','$\ \ \ \textnormal{PathCtrl}$'},'Interpreter','latex')
    ylabel({'\textnormal{Percentage}'},'Interpreter','latex')
    xlabel({'$\textnormal{Control cost}$'},'Interpreter','latex')
    %---------------------------------------------------------------------------------------------------%
    
    %---------------------------------------------------------------------------------------------------%
    % (f) comparison between RobDetCtrl and PathCtrl for (best and worst) coefficient realizations
    %---------------------------------------------------------------------------------------------------%
    figure('NumberTitle','off','Name','rob_det_ctrl v.s. path_ctrl using KLE-normal: comparison results');
    
    index_target_mismatch = rob_det_ctrl_TargetMatch_Index(1:4);
    num = size(rob_det_ctrl_OptState,2);
    index_target_match = rob_det_ctrl_TargetMatch_Index(num-3:num);
    
    fprintf('\n')
    fprintf(' Target Match for Traindata: coefficient index (worst case) for RobDetCtrl = %d',index_target_mismatch(1));
    fprintf('\n')
    fprintf(' Target Match for Traindata: coefficient index (best case) for RobDetCtrl = %d',index_target_match(4));
    fprintf('\n')
    
    % coeff - worst case -  rob-det-ctrl
    subplot(4,4,1)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',train_data_inputs_Npts(:,index_target_mismatch(1)));
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
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',train_data_inputs_Npts(:,index_target_match(4)));
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
    
    % OptCtrl - rob-det-ctrl
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
    
    % OptCtrl - pathwise ctrl - worst case coefficient
    subplot(4,4,7)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',path_ctrl_OptCtrl(:,index_target_mismatch(1)));
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
    
    % OptCtrl - pathwise ctrl - best case coefficient
    subplot(4,4,8)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',path_ctrl_OptCtrl(:,index_target_match(4)));
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
    
    % OptState - rob-det-ctrl - worst case coefficient
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
    
    % OptState - rob-det-ctrl - best case coefficient
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
    
    % OptState - path-ctrl - worst case coefficient
    subplot(4,4,11)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',path_ctrl_OptState(:,index_target_mismatch(1)));
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
    
    % OptState - path-ctrl - best case coefficient
    subplot(4,4,12)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',path_ctrl_OptState(:,index_target_match(4)));
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
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',path_ctrl_OptState(:,index_target_mismatch(1)) - U(V_basis(1,:),V_basis(2,:))');
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
    
    % state err - best case -  path-ctrl
    subplot(4,4,16)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',path_ctrl_OptState(:,index_target_match(4)) - U(V_basis(1,:),V_basis(2,:))');
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
    %---------------------------------------------------------------------------------------------------%
    
end


%% auxiliray figures (activate if interested, not required for paper)
%
%
%     %---------------------------------------------------------------------------------------------------%
%     % (a) iterative objective values and gradient errors
%     %---------------------------------------------------------------------------------------------------%
%     figure('NumberTitle','off','Name','rob_det_ctrl v.s. path_ctrl using KLE coeff: iterative objective values and gradient errors');
%
%     subplot(1,2,1)
%     plot(1:size(rob_det_ctrl_IteObj,2),rob_det_ctrl_IteObj(2,:),'-r')
%     hold on
%     plot(1:size(path_ctrl_IteObj,2),path_ctrl_IteObj(2,:),'-b')
%     hold on
%     legend({'$\ \textnormal{robust control}\ \hat{f}^{[k]}_{h,n}$','$\ \textnormal{pathwise control}\ f_{\omega,h}^{[k]}$'},'Interpreter','latex')
%     set(gcf,'color','w')
%     set(gca,'FontSize',18);
%     pbaspect([7 4 1])
%     xlabel({'index $k$ of iteration'},'Interpreter','latex')
%     ylabel({'$\mathcal{J}_{\textnormal{\fontsize{8}{0}\selectfont MC-FE}}(f)$'},'Interpreter','latex')
%     hold on
%
%     subplot(1,2,2)
%     X = rob_det_ctrl_IteErr(1,:);
%     Y_Maxerr_L2 = log(rob_det_ctrl_IteErr(3,:));
%     Maxerr_L2 = polyfit(X,Y_Maxerr_L2,1);
%     p_Maxerr_L2 = polyval(Maxerr_L2,X);
%     plot(X,p_Maxerr_L2,'-r')
%     hold on
%     X = path_ctrl_IteErr(1,:);
%     Y_Maxerr_L2 = log(path_ctrl_IteErr(3,:));
%     Maxerr_L2 = polyfit(X,Y_Maxerr_L2,1);
%     p_Maxerr_L2 = polyval(Maxerr_L2,X);
%     plot(X,p_Maxerr_L2,'-b')
%     hold on
%     legend({'$\ \textnormal{robust control}\ \hat{f}^{[k]}_{h,n}$','$\ \textnormal{pathwise control}\ f_{\omega,h}^{[k]}$'},'Interpreter','latex')
%     set(gcf,'color','w')
%     set(gca,'FontSize',18);
%     pbaspect([7 4 1])
%     xlabel({'index $k$ of iteration'},'Interpreter','latex')
%     ylabel({'$\log \| \nabla \mathcal{J}_{\textnormal{\fontsize{8}{0}\selectfont MC-FE}}(f) \|_{L^2(D)}$'},'Interpreter','latex')
%     %---------------------------------------------------------------------------------------------------%
%
%
%     %---------------------------------------------------------------------------------------------------%
%     % (b) target, optimal state and control variables
%     %---------------------------------------------------------------------------------------------------%
%     figure('NumberTitle','off','Name','rob_det_ctrl v.s. path_ctrl using KLE coeff: target function, mean of optimal states, and (mean of) optimal control');
%
%     subplot(2,3,1)
%     Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',U(V_basis(1,:),V_basis(2,:))');
%     qz=Ft(qx,qy);
%     imagesc(qz)
%     colorbar off
%     caxis([-2 2])
%     colormap jet
%     axis xy
%     ax = gca;
%     axis(ax,'off')
%     xlabel(ax,{'$U(x)$'},'Interpreter','latex');
%     set(get(ax,'XLabel'),'Visible','on')
%     axis equal
%     set(gca,'FontSize',18);
%     set(gcf,'color','w')
%     rectangle('position',[0 0 1 1] )
%     xh = get(gca,'xlabel'); % handle to the label object
%     p = get(xh,'position'); % get the current position property
%     p(2) = 0.1 * p(2) ;
%     set(xh,'position',p);
%     hold on
%
%     subplot(2,3,2)
%     Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',mean(rob_det_ctrl_OptState,2));
%     qz=Ft(qx,qy);
%     imagesc(qz)
%     colorbar off
%     caxis([-2 2])
%     colormap jet
%     axis xy
%     ax = gca;
%     axis(ax,'off')
%     xlabel({'$E_n[\hat{u}^*_{h}(x,\omega)]$'},'Interpreter','latex');
%     set(get(ax,'XLabel'),'Visible','on')
%     axis equal
%     set(gca,'FontSize',18);
%     set(gcf,'color','w')
%     rectangle('position',[0 0 1 1] )
%     xh = get(gca,'xlabel'); % handle to the label object
%     p = get(xh,'position'); % get the current position property
%     p(2) = 0.1 * p(2) ;        % double the distance,
%     set(xh,'position',p);
%     hold on
%
%     subplot(2,3,3)
%     Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',rob_det_ctrl_OptCtrl(:));
%     qz=Ft(qx,qy);
%     imagesc(qz)
%     colorbar off
%     colormap jet
%     axis xy
%     ax = gca;
%     axis(ax,'off')
%     xlabel({'$\hat{f}^*_{h,n}(x)$'},'Interpreter','latex');
%     set(get(ax,'XLabel'),'Visible','on')
%     axis equal
%     set(gca,'FontSize',18);
%     set(gcf,'color','w')
%     rectangle('position',[0 0 1 1] )
%     xh = get(gca,'xlabel'); % handle to the label object
%     p = get(xh,'position'); % get the current position property
%     p(2) = 0.1 * p(2) ; % double the distance,
%     set(xh,'position',p);
%
%     subplot(2,3,4)
%     Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',U(V_basis(1,:),V_basis(2,:))');
%     qz=Ft(qx,qy);
%     imagesc(qz)
%     colorbar off
%     caxis([-2 2])
%     colormap jet
%     axis xy
%     ax = gca;
%     axis(ax,'off')
%     xlabel(ax,{'$U(x)$'},'Interpreter','latex');
%     set(get(ax,'XLabel'),'Visible','on')
%     axis equal
%     set(gca,'FontSize',18);
%     set(gcf,'color','w')
%     rectangle('position',[0 0 1 1] )
%     xh = get(gca,'xlabel'); % handle to the label object
%     p = get(xh,'position'); % get the current position property
%     p(2) = 0.1 * p(2) ;
%     set(xh,'position',p);
%     hold on
%
%     subplot(2,3,5)
%     Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',mean(path_ctrl_OptState,2));
%     qz=Ft(qx,qy);
%     imagesc(qz)
%     colorbar off
%     caxis([-2 2])
%     colormap jet
%     axis xy
%     ax = gca;
%     axis(ax,'off')
%     xlabel({'$E_n[u^*_{\omega,h}(x)]$'},'Interpreter','latex');
%     set(get(ax,'XLabel'),'Visible','on')
%     axis equal
%     set(gca,'FontSize',18);
%     set(gcf,'color','w')
%     rectangle('position',[0 0 1 1] )
%     xh = get(gca,'xlabel'); % handle to the label object
%     p = get(xh,'position'); % get the current position property
%     p(2) = 0.1 * p(2) ;        % double the distance,
%     set(xh,'position',p);
%     hold on
%
%     subplot(2,3,6)
%     Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',mean(path_ctrl_OptCtrl,2));
%     qz=Ft(qx,qy);
%     imagesc(qz)
%     colorbar off
%     colormap jet
%     axis xy
%     ax = gca;
%     axis(ax,'off')
%     xlabel({'$E_n[f^*_{\omega,h}(x)]$'},'Interpreter','latex');
%     set(get(ax,'XLabel'),'Visible','on')
%     axis equal
%     set(gca,'FontSize',18);
%     set(gcf,'color','w')
%     rectangle('position',[0 0 1 1] )
%     xh = get(gca,'xlabel'); % handle to the label object
%     p = get(xh,'position'); % get the current position property
%     p(2) = 0.1 * p(2) ; % double the distance,
%     set(xh,'position',p);
%     %---------------------------------------------------------------------------------------------------%
%
%
%     %---------------------------------------------------------------------------------------------------%
%     % (d) worst-4 target match using rob_det_ctrl
%     %---------------------------------------------------------------------------------------------------%
%     figure('NumberTitle','off','Name','rob_det_ctrl v.s. path_ctrl using KLE coeff: target mismatch');
%
%     index_target_mismatch = rob_det_ctrl_TargetMatch_Index(1:4);
%
%     for i = 1 : 4
%
%         subplot(3,4,i)
%         Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',train_data_inputs_Npts(:,index_target_mismatch(i)));
%         qz = Ft(qx,qy);
%         imagesc(qz)
%         colormap jet
%         set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
%         set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
%         colorbar off
%         axis xy
%         axis off
%         axis equal
%         set(gca,'FontSize',18);
%         set(gcf,'color','w')
%         hold on
%         rectangle('position',[0 0 1 1] )
%         ylabel({'$a(x,\omega)$'},'Interpreter','latex')
%         %title({['state No.' num2str(index_target_mismatch(i))]});
%         hold on
%
%         subplot(3,4,4+i)
%         Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',rob_det_ctrl_OptState(:,index_target_mismatch(i)));
%         qz = Ft(qx,qy);
%         imagesc(qz)
%         colormap jet
%         set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
%         set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
%         colorbar off
%         % caxis([-2 2]) % used for state only
%         axis xy
%         axis off
%         axis equal
%         set(gca,'FontSize',18);
%         set(gcf,'color','w')
%         hold on
%         rectangle('position',[0 0 1 1] )
%         ylabel({'$u^*_h(x,\omega)$'},'Interpreter','latex')
%         %title({['state No.' num2str(index_target_mismatch_rob_ctrl_KLE_coeff(i))]});
%         hold on
%
%         subplot(3,4,8+i)
%         Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',path_ctrl_OptState(:,index_target_mismatch(i)));
%         qz = Ft(qx,qy);
%         imagesc(qz)
%         colormap jet
%         set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
%         set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
%         colorbar off
%         caxis([-2 2]) % used for state only
%         axis xy
%         axis off
%         axis equal
%         set(gca,'FontSize',18);
%         set(gcf,'color','w')
%         hold on
%         rectangle('position',[0 0 1 1] )
%         ylabel({'$u^*_{\omega,h}(x)$'},'Interpreter','latex')
%         %title({['state No.' num2str(index_target_mismatch_rob_ctrl_KLE_coeff(i))]});
%         hold on
%
%     end
%     %---------------------------------------------------------------------------------------------------%
%
%
%     %---------------------------------------------------------------------------------------------------%
%     % (e) best-4 target match using rob_det_ctrl
%     %---------------------------------------------------------------------------------------------------%
%     figure('NumberTitle','off','Name','rob_det_ctrl v.s. path_ctrl using KLE coeff: target match');
%
%     num = size(rob_det_ctrl_OptState,2);
%     index_target_match = rob_det_ctrl_TargetMatch_Index(num-3:num);
%
%     for i = 1 : 4
%
%         subplot(3,4,i)
%         Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',train_data_inputs_Npts(:,index_target_match(i)));
%         qz = Ft(qx,qy);
%         imagesc(qz)
%         colormap jet
%         set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
%         set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
%         colorbar off
%         axis xy
%         axis off
%         axis equal
%         set(gca,'FontSize',18);
%         set(gcf,'color','w')
%         hold on
%         rectangle('position',[0 0 1 1] )
%         ylabel({'$a(x,\omega)$'},'Interpreter','latex')
%         %title({['state No.' num2str(index_target_mismatch(i))]});
%         hold on
%
%         subplot(3,4,4+i)
%         Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',rob_det_ctrl_OptState(:,index_target_match(i)));
%         qz = Ft(qx,qy);
%         imagesc(qz)
%         colormap jet
%         set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
%         set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
%         colorbar off
%         caxis([-2 2]) % used for state only
%         axis xy
%         axis off
%         axis equal
%         set(gca,'FontSize',18);
%         set(gcf,'color','w')
%         hold on
%         rectangle('position',[0 0 1 1] )
%         ylabel({'$u^*_h(x,\omega)$'},'Interpreter','latex')
%         %title({['state No.' num2str(index_target_mismatch_rob_ctrl_KLE_coeff(i))]});
%         hold on
%
%         subplot(3,4,8+i)
%         Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',path_ctrl_OptState(:,index_target_match(i)));
%         qz = Ft(qx,qy);
%         imagesc(qz)
%         colormap jet
%         set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
%         set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
%         colorbar off
%         caxis([-2 2]) % used for state only
%         axis xy
%         axis off
%         axis equal
%         set(gca,'FontSize',18);
%         set(gcf,'color','w')
%         hold on
%         rectangle('position',[0 0 1 1] )
%         ylabel({'$u^*_{\omega,h}(x)$'},'Interpreter','latex')
%         %title({['state No.' num2str(index_target_mismatch_rob_ctrl_KLE_coeff(i))]});
%         hold on
%
%     end
%     %---------------------------------------------------------------------------------------------------%


end




