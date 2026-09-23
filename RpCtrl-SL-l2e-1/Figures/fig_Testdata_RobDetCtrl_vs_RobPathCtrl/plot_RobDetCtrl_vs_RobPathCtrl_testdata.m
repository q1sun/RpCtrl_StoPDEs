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
    U_vals = U(V_basis(1,:), V_basis(2,:))'; 
        
    num = size(rob_det_ctrl_OptState,2);
    index_target_match = rob_det_ctrl_TargetMatch_Index_testdata(num-3:num);
    index_target_mismatch = rob_det_ctrl_TargetMatch_Index_testdata(1:4);
    idx_worst = index_target_mismatch(1); 
    idx_best  = index_target_match(1);    

    err_det_worst  = rob_det_ctrl_OptState(:,idx_worst) - U_vals;
    err_path_worst = rob_path_ctrl_OptState(:,idx_worst) - U_vals;
    err_det_best   = rob_det_ctrl_OptState(:,idx_best) - U_vals;
    err_path_best  = rob_path_ctrl_OptState(:,idx_best) - U_vals;
    
    fprintf('\n')
    fprintf(' Target Match for Testdata: coefficient index (worst case) for RobDetCtrl = %d',idx_worst);
    fprintf('\n')
    fprintf(' Target Match for Testdata: coefficient index (best case) for RobDetCtrl = %d',idx_best);
    fprintf('\n')

    %------------------%
    fig1 = figure('NumberTitle','off','Name',['Worst Case: Index ', num2str(idx_worst)],'Position',[50 100 1600 700],'color','w');
    
    % a_{h,omega_worst}(x)
    subplot(2,4,[1, 5])
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',test_data_inputs_Npts(:,idx_worst));
    imagesc(Ft(qx,qy)); colormap jet; axis xy equal tight off;
    set(gca,'FontSize',23); hold on; rectangle('position',[0 0 Lx Ly]);
    title(['$a_{h,\,\omega_{', num2str(idx_worst), '}}(x)$'],'Interpreter','latex','FontSize',23);
    colorbar; 
    % robust deterministic control
    subplot(2,4,2)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',rob_det_ctrl_OptCtrl(:));
    imagesc(Ft(qx,qy)); colormap jet; axis xy equal tight off;
    set(gca,'FontSize',23); hold on; rectangle('position',[0 0 Lx Ly]);
    title('$\hat{f}_h^*(x)$','Interpreter','latex','FontSize',23);
    colorbar;
    % 3. state
    subplot(2,4,3)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',rob_det_ctrl_OptState(:,idx_worst));
    imagesc(Ft(qx,qy)); colormap jet; axis xy equal tight off; 
    set(gca,'FontSize',23); hold on; rectangle('position',[0 0 Lx Ly]);
    title(['$\hat{u}_{h,\,\omega_{', num2str(idx_worst), '}}^*(x)$'],'Interpreter','latex','FontSize',23);
    colorbar;
    % 4. target match error
    subplot(2,4,4)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',err_det_worst);
    imagesc(Ft(qx,qy)); colormap jet; axis xy equal tight off; 
    set(gca,'FontSize',23); hold on; rectangle('position',[0 0 Lx Ly]);
    title(['$\hat{u}_{h,\,\omega_{', num2str(idx_worst), '}}^*(x) - U(x)$'],'Interpreter','latex','FontSize',23);
    colorbar;
    % 5. data-driven robust-pathwise control
    subplot(2,4,6)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',rob_path_ctrl_OptCtrl(:,idx_worst));
    imagesc(Ft(qx,qy)); colormap jet; axis xy equal tight off;
    set(gca,'FontSize',23); hold on; rectangle('position',[0 0 Lx Ly]);
    title(['$f^{\,\theta^*}_{h,\,\omega_{', num2str(idx_worst), '}}(x)$'],'Interpreter','latex','FontSize',23);
    colorbar;
    % 6. state
    subplot(2,4,7)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',rob_path_ctrl_OptState(:,idx_worst));
    imagesc(Ft(qx,qy)); colormap jet; axis xy equal tight off; % 移除了 caxis
    set(gca,'FontSize',23); hold on; rectangle('position',[0 0 Lx Ly]);
    title(['$u^*_{h,\,\omega_{', num2str(idx_worst), '}}(x)$'],'Interpreter','latex','FontSize',23);
    colorbar;
    % 7. target match error
    subplot(2,4,8)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',err_path_worst);
    imagesc(Ft(qx,qy)); colormap jet; axis xy equal tight off; 
    set(gca,'FontSize',23); hold on; rectangle('position',[0 0 Lx Ly]);
    title(['$u^*_{h,\,\omega_{', num2str(idx_worst), '}}(x) - U(x)$'],'Interpreter','latex','FontSize',23);
    colorbar;

    % --- Figure 1: Colorbar Alignment ---
    drawnow; 
    axs = findobj(fig1, 'Type', 'axes');
    for i = 1:length(axs)
        ax = axs(i); cb = ax.Colorbar;
        if ~isempty(cb)
            original_units = ax.Units; ax.Units = 'pixels'; pos = ax.Position; 
            pbar = ax.PlotBoxAspectRatio; img_ratio = pbar(1) / pbar(2); box_ratio = pos(3) / pos(4);
        
            if box_ratio > img_ratio
                true_h = pos(4); true_w = true_h * img_ratio;
                true_l = pos(1) + (pos(3) - true_w) / 2; true_b = pos(2);
            else
                true_w = pos(3); true_h = true_w / img_ratio;
                true_l = pos(1); true_b = pos(2) + (pos(4) - true_h) / 2;
            end
            ax.Position = pos; 
            fig_pos = fig1.Position; cb_w_norm = 12 / fig_pos(3); cb_gap_norm = 10 / fig_pos(3); 
        
            cb.Position = [ (true_l + true_w)/fig_pos(3) + cb_gap_norm, true_b / fig_pos(4), cb_w_norm, true_h / fig_pos(4) ];                          
            ax.Units = original_units; 
        end
    end

    ax = gcf;
    exportgraphics(ax, 'Figures//fig_Testdata_RobDetCtrl_vs_RobPathCtrl//fig_testdata_RobDetCtrl_vs_RobPathCtrl_worst_SL.pdf')

    %------------------%
    fig2 = figure('NumberTitle','off','Name',['Best Case: Index ', num2str(idx_best)],'Position',[100 150 1600 700],'color','w');
    
    % 1. a_{h,omega_best}(x)
    subplot(2,4,[1, 5])
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',test_data_inputs_Npts(:,idx_best));
    imagesc(Ft(qx,qy)); colormap jet; axis xy equal tight off;
    set(gca,'FontSize',23); hold on; rectangle('position',[0 0 Lx Ly]);
    title(['$a_{h,\,\omega_{', num2str(idx_best), '}}(x)$'],'Interpreter','latex','FontSize',23);
    colorbar; 
    % 2. robust deterministic control
    subplot(2,4,2)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',rob_det_ctrl_OptCtrl(:));
    imagesc(Ft(qx,qy)); colormap jet; axis xy equal tight off;
    set(gca,'FontSize',23); hold on; rectangle('position',[0 0 Lx Ly]);
    title('$\hat{f}_h^*(x)$','Interpreter','latex','FontSize',23);
    colorbar;
    % 3. state
    subplot(2,4,3)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',rob_det_ctrl_OptState(:,idx_best));
    imagesc(Ft(qx,qy)); colormap jet; axis xy equal tight off; % 移除了 caxis
    set(gca,'FontSize',23); hold on; rectangle('position',[0 0 Lx Ly]);
    title(['$\hat{u}_{h,\,\omega_{', num2str(idx_best), '}}^*(x)$'],'Interpreter','latex','FontSize',23);
    colorbar;
    % 4. target match error
    subplot(2,4,4)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',err_det_best);
    imagesc(Ft(qx,qy)); colormap jet; axis xy equal tight off; 
    set(gca,'FontSize',23); hold on; rectangle('position',[0 0 Lx Ly]);
    title(['$\hat{u}_{h,\,\omega_{', num2str(idx_best), '}}^*(x) - U(x)$'],'Interpreter','latex','FontSize',23);
    colorbar;
    % 5. data-driven robust-pathwise control
    subplot(2,4,6)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',rob_path_ctrl_OptCtrl(:,idx_best));
    imagesc(Ft(qx,qy)); colormap jet; axis xy equal tight off;
    set(gca,'FontSize',23); hold on; rectangle('position',[0 0 Lx Ly]);
    title(['$f^{\,\theta^*}_{h,\,\omega_{', num2str(idx_best), '}}(x)$'],'Interpreter','latex','FontSize',23);
    colorbar;
    % 6. state
    subplot(2,4,7)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',rob_path_ctrl_OptState(:,idx_best));
    imagesc(Ft(qx,qy)); colormap jet; axis xy equal tight off; % 移除了 caxis
    set(gca,'FontSize',23); hold on; rectangle('position',[0 0 Lx Ly]);
    title(['$u^*_{h,\,\omega_{', num2str(idx_best), '}}(x)$'],'Interpreter','latex','FontSize',23);
    colorbar;
    % 7. target match error
    subplot(2,4,8)
    Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',err_path_best);
    imagesc(Ft(qx,qy)); colormap jet; axis xy equal tight off; 
    set(gca,'FontSize',23); hold on; rectangle('position',[0 0 Lx Ly]);
    title(['$u^*_{h,\,\omega_{', num2str(idx_best), '}}(x) - U(x)$'],'Interpreter','latex','FontSize',23);
    colorbar;

    % --- Figure 2: Colorbar Alignment ---
    drawnow;
    axs = findobj(fig2, 'Type', 'axes');
    for i = 1:length(axs)
        ax = axs(i); cb = ax.Colorbar;
        if ~isempty(cb)
            original_units = ax.Units; ax.Units = 'pixels'; pos = ax.Position;
            pbar = ax.PlotBoxAspectRatio; img_ratio = pbar(1) / pbar(2); box_ratio = pos(3) / pos(4);

            if box_ratio > img_ratio
                true_h = pos(4); true_w = true_h * img_ratio;
                true_l = pos(1) + (pos(3) - true_w) / 2; true_b = pos(2);
            else
                true_w = pos(3); true_h = true_w / img_ratio;
                true_l = pos(1); true_b = pos(2) + (pos(4) - true_h) / 2;
            end
            ax.Position = pos;
            fig_pos = fig2.Position; cb_w_norm = 12 / fig_pos(3); cb_gap_norm = 10 / fig_pos(3);

            cb.Position = [ (true_l + true_w)/fig_pos(3) + cb_gap_norm, true_b / fig_pos(4), cb_w_norm, true_h / fig_pos(4) ];
            ax.Units = original_units;
        end
    end

    ax = gcf;
    exportgraphics(ax, 'Figures//fig_Testdata_RobDetCtrl_vs_RobPathCtrl//fig_testdata_RobDetCtrl_vs_RobPathCtrl_best_SL.pdf')    
end


end




