function plot_KLE_eigenpairs(plt_KLE_eigenpair)

format short e

addpath('Figures/fig_KLE')
load('Figures/fig_KLE/Fig_Data_KLE.mat');

if plt_KLE_eigenpair == 1
    
    %%%%%%%%%%%%%%%
    d = min(40,dim_TKLE_log_a);
    figure('NumberTitle','off','Name','KLE of log-coeff : eigenvalues and energy ratio','Position', [100, 100, 1600, 400]);
    
    subplot(1,2,1)
    plot(1:d,lambda(1:d),'-bo',...
        'LineWidth',1,...
        'MarkerEdgeColor','b',...
        'MarkerFaceColor','b',...
        'MarkerSize',5);
    %title({'decay of eigenvalue w.r.t. $\textnormal{Cov}_a(x,y)$'},'Interpreter','latex');
    xlabel('index $m$ of eigenvalue','Interpreter','latex');
    ylabel('eigenvalue $\lambda_m^h$','Interpreter','latex');
    xlim([1 d])
    set(gcf,'color','w')
    set(gca,'FontSize',23);

    subplot(1,2,2)
    plot(1:d,energy_ratio(1,1:d),'-ro',...
        'LineWidth',1,...
        'MarkerEdgeColor','r',...
        'MarkerFaceColor','r',...
        'MarkerSize',5);
    %title({'cumulative energy ratio w.r.t. $\textnormal{Cov}_a(x,y)$'},'Interpreter','latex');
    xlabel('total number $M$ of cumulative eigenvalues','Interpreter','latex');
    ylabel('energy ratio $\rho_M$','Interpreter','latex');
    xlim([1 d])
    set(gcf,'color','w')
    set(gca,'FontSize',23);
    ylim([0 1])

    yt = yticks;
    yticklabels(yt * 100);
    yticklabels(strcat(string(yt * 100), '%'));

    ax = gcf;
    exportgraphics(ax, 'Figures//fig_KLE//fig_KLE_eigenvalues.pdf')
    %%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%
    Lx = max(V_basis(1,:));
    Ly = max(V_basis(2,:));
    dx = 0:.002:Lx;
    dy = 0:0.002:Ly;
    [qx,qy] = meshgrid(dx,dy);
    
    figure('NumberTitle','off','Name','Coeff KLE: eigenfunctions');
    for i = 1 : 10
        
        subplot(2,5,i)
        Ft = TriScatteredInterp(V_basis(1,:)',V_basis(2,:)',phi(:,i));
        qz = Ft(qx,qy);
        imagesc(qz)
        colormap jet
        set(gca,'xtick',[1 length(dx)],'xticklabel',[0 Lx])
        set(gca,'ytick',[1 length(dy)],'yticklabel',[0 Ly])
        colorbar off
        axis xy
        title({['No.' num2str(i) ' eigenfunction']});
        set(gcf,'color','w')
        hold on
        
    end
    %%%%%%%%%%%%%%%    
end

end