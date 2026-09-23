function plot_RobPathCtrl_vs_PathCtrl_traindata(plt_path_vs_robpath_traindata)

format short e

% load('Checkpoints/RobDetCtrl_KLE_Normal.mat')
load('Figures/fig_Traindata_RobPathCtrl_vs_PathCtrl/Fig_Traindata_PathCtrl_KLE_Normal.mat');
load('Figures/fig_Traindata_RobPathCtrl_vs_PathCtrl/Fig_Traindata_RobPathCtrl_KLE_Normal_Trained_Model.mat');

% opengl('software')
% opengl('save','software')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate index for plot
% dif =  rob_path_ctrl_OptCtrl_trainbatch_sim2 - repmat(rob_det_ctrl_KLE_normal,1,size(rob_path_ctrl_OptCtrl_trainbatch_sim2,2));
dif =  rob_path_ctrl_OptCtrl_trainbatch - path_ctrl_OptCtrl(:,1:size(rob_path_ctrl_OptCtrl_trainbatch,2));
for i = 1 : size(rob_path_ctrl_OptCtrl_trainbatch,2)
    dif_norm(1,i) = norm(dif(:,i),1);
end
[~,index] = sort(dif_norm,'descend'); % if not satisfied, can modify the choice of index manually,
% e.g., index = [index(8),index(47),index(94)];
index = [index(9),index(90),index(48)];

% plot coeff, ground-truth, and prediction
if plt_path_vs_robpath_traindata == 1

    figure('NumberTitle','off','Name','training results: path_ctrl v.s. rob_path_ctrl using KLE-normal', ...
        'Position', [100, 100, 1300, 900], 'Color', 'w');
    t = tiledlayout(3, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

    sgtitle(' ', 'FontSize', 28);

    Lx = max(V_basis(1,:));
    Ly = max(V_basis(2,:));
    dx = 0:0.002:Lx;
    dy = 0:0.002:Ly;
    [qx,qy] = meshgrid(dx,dy);

    for i = 1 : 3

        % ================= column 1 =================
        nexttile(4*i-3)
        Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',train_data_inputs_Npts(:,index(i)));
        qz = Ft(qx,qy);
        imagesc(qz)
        colormap jet
        set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
        set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
        colorbar
        axis xy
        axis image
        axis off
        set(gca,'FontSize',23);
        hold on
        rectangle('position',[0 0 1 1] )
        if i == 1
            title('$a_{h,\omega}(x)$', 'FontSize', 23, 'Interpreter', 'latex');
        end

        % ================= column 2 =================
        nexttile(4*i-2)
        Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',path_ctrl_OptCtrl(:,index(i)));
        qz = Ft(qx,qy);
        imagesc(qz)
        colormap jet
        set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
        set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
        colorbar
        axis xy
        axis image
        axis off
        set(gca,'FontSize',23);
        hold on
        rectangle('position',[0 0 1 1] )
        if i == 1
            title('$f_{h,\omega}^*(x)$', 'FontSize', 23, 'Interpreter', 'latex');
        end

        % ================= column 3 =================
        nexttile(4*i-1)
        Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',rob_path_ctrl_OptCtrl_trainbatch(:,index(i)));
        qz = Ft(qx,qy);
        imagesc(qz)
        colormap jet
        set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
        set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
        colorbar
        axis xy
        axis image
        axis off
        set(gca,'FontSize',23);
        hold on
        rectangle('position',[0 0 1 1] )
        if i == 1
            title('$f_{h,\omega}^{\,\theta^*}(x)$', 'FontSize', 23, 'Interpreter', 'latex');
        end

        % ================= column 4 =================
        nexttile(4*i)
        Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',path_ctrl_OptCtrl(:,index(i)) - rob_path_ctrl_OptCtrl_trainbatch(:,index(i)));
        qz = Ft(qx,qy);
        imagesc(qz)
        colormap jet
        set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
        set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
        colorbar
        axis xy
        axis image
        axis off
        set(gca,'FontSize',23);
        hold on
        rectangle('position',[0 0 1 1] )
        if i == 1
            title('$f_{h,\omega}^*(x) - f_{h,\omega}^{\,\theta^*}(x)$', 'FontSize', 23, 'Interpreter', 'latex');
        end
    end

    exportgraphics(gcf, 'Figures//fig_Traindata_RobPathCtrl_vs_PathCtrl//fig_train_results_SL.pdf', 'ContentType', 'vector', 'BackgroundColor', 'w');

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
index = [1368,3372]; % worst and best cases

if size(rob_path_ctrl_OptCtrl_trainbatch,2) < min(index)
    fprintf(' Warning! Not network predictions for the worst and best cases!')
    fprintf('\n')
    fprintf(' (check coefficient indices in plot_RobPathCtrl_vs_PathCtrl_traindata.m)')
    fprintf('\n')

else
    figure('NumberTitle','off','Name','training restuls (worst and best coeffs): path_ctrl v.s. rob_path_ctrl using KLE-normal');

    Lx = max(V_basis(1,:));
    Ly = max(V_basis(2,:));
    dx = 0:0.002:Lx;
    dy = 0:0.002:Ly;
    [qx,qy] = meshgrid(dx,dy);

    for i = 1 : 2

        subplot(2,3,3*i-2)
        Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',train_data_inputs_Npts(:,index(i)));
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

        subplot(2,3,3*i-1)
        Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',path_ctrl_OptCtrl(:,index(i)));
        qz = Ft(qx,qy);
        imagesc(qz)
        colormap jet
        set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
        set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
        colorbar off
        % caxis([-2 2]) % used for state only
        axis xy
        axis off
        axis equal
        set(gca,'FontSize',18);
        set(gcf,'color','w')
        hold on
        rectangle('position',[0 0 1 1] )
        ylabel({'$u^*_h(x,\omega)$'},'Interpreter','latex')
        %title({['state No.' num2str(index_target_mismatch_rob_ctrl_KLE_coeff(i))]});
        hold on

        subplot(2,3,3*i)
        Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',rob_path_ctrl_OptCtrl_trainbatch_sim2(:,index(i)));
        qz = Ft(qx,qy);
        imagesc(qz)
        colormap jet
        set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
        set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
        colorbar off
        %caxis([-2 2]) % used for state only
        axis xy
        axis off
        axis equal
        set(gca,'FontSize',18);
        set(gcf,'color','w')
        hold on
        rectangle('position',[0 0 1 1] )
        ylabel({'$u^*_{\omega,h}(x)$'},'Interpreter','latex')
        %title({['state No.' num2str(index_target_mismatch_rob_ctrl_KLE_coeff(i))]});
        hold on

    end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end




