function [V_linear,T_linear,BN_linear_Dirichlet,V_quadratic,T_quadratic,BN_quadratic_Dirichlet,BE_linear_Dirichlet,BE_quadratic_Dirichlet] = generate_mesh_linear_quadratic_element(h,plot_mesh_linear)

% The problem domain is a rectangle [left,right]*[bottom,top]. However we can only do the tessellation for regular shaped domain.
   left=0;
   right=1;
   bottom=0;
   top=1;

% plot mesh controller
%    plot_mesh_linear=0;
   plot_mesh_quadratic=0;
   
% Remark: I have changed the order for BE_linear_Dirichlet and BE_quadratic_Dirichlet so that one can use them for command 'pdetool'.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate mesh information for linear element   
   N1_linear=(right-left)/h(1);%N1 is the number of the sub-intervals of the partition in x-direction.
   N2_linear=(top-bottom)/h(2);%N2 is the number of the sub-intervals of the partition in y-direction.
   tnp_linear=(N1_linear+1)*(N2_linear+1);%tnp:total number of all the nodes of FE,including inner nodes and boundary nodes.
   V_linear=zeros(2,tnp_linear);%M stores the coordinates of all nodes for the type of finite element specified by "basis_type", M(i,j) is the ith coordinate of the jth node.
   T_linear=zeros(3,2*N1_linear*N2_linear);%T stores the global indices of the nodes of every element for the type of finite element specified by "basis_type", T(i,j) stores the global index of the ith node in th jth element.
   Q_linear=zeros(N1_linear+1,N2_linear+1);

   for j=1:tnp_linear
      if mod(j,N2_linear+1)==0
         V_linear(1,j)=left+(j/(N2_linear+1)-1)*h(1);
         V_linear(2,j)=top;
      else
         V_linear(1,j)=left+fix(j/(N2_linear+1))*h(1);
         V_linear(2,j)=bottom+(mod(j,N2_linear+1)-1)*h(2);
      end
   end

   for i=1:N1_linear+1
      for j=1:N2_linear+1
         Q_linear(i,j)=(i-1)*(N2_linear+1)+j;
      end
   end

%Go through all rectangles in the partition. 
%For the nth rectangle, store the information of its two triangular elements whose element indices are 2n-1 and 2n.
   for n=1:N1_linear*N2_linear
   
      if mod(n,N2_linear)==0
         row=N2_linear;
         column=n/N2_linear;
      else
         row=mod(n,N2_linear);
         column=fix(n/N2_linear)+1;
      end
   
      T_linear(1,2*n-1)=Q_linear(column,row);
      T_linear(2,2*n-1)=Q_linear(column+1,row);
      T_linear(3,2*n-1)=Q_linear(column,row+1);  
  
      T_linear(1,2*n)=Q_linear(column,row+1);
      T_linear(2,2*n)=Q_linear(column+1,row);
      T_linear(3,2*n)=Q_linear(column+1,row+1);  
    
   end
   

   if plot_mesh_linear==1
       figure('NumberTitle','off','Name',['linear element: mesh with h = ' num2str(h(1))]);
       patch('Faces',T_linear','Vertices',V_linear','FaceColor','w');
       hold on
       plot(V_linear(1,:),V_linear(2,:),'*');
       text(V_linear(1,:),V_linear(2,:),num2str((1:size(V_linear,2))'),'FontSize',12,'Color','red');
       % order of face displayed in the mean value coordinates of the triangle
       sizeT_linear=size(T_linear);
       T1_linear=zeros(3,size(T_linear,2));
       T2_linear=zeros(3,size(T_linear,2));
       for i=1:sizeT_linear(1)
           for j=1:sizeT_linear(2)
           T1_linear(i,j)=V_linear(1,T_linear(i,j));
           T2_linear(i,j)=V_linear(2,T_linear(i,j));
           end
       end
       Txx_linear=mean(T1_linear);
       Tyy_linear=mean(T2_linear);
       text(Txx_linear(:),Tyy_linear(:),num2str((1:size(T_linear,2))'),'FontSize',10,'HorizontalAlignment','left','BackgroundColor',[.7 .9 .7]);     
       % some parameters
       axis([min(V_linear(1,:))-0.02 max(V_linear(1,:))+0.02 min(V_linear(2,:))-0.02 max(V_linear(2,:))+0.02]);
       title('mesh for linear element');
       xlabel('x label');
       ylabel('y label');
       hold off
   end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%generate boundary node information for linear element
nbn_linear=2*(N1_linear+N2_linear);
BN_linear_Dirichlet=zeros(2,nbn_linear);
%all Dirichlet boundary nodes
BN_linear_Dirichlet(1,:)=-1;
%bottom boundary nodes.
for k=1:N1_linear
    BN_linear_Dirichlet(2,k)=(k-1)*(N2_linear+1)+1;
end

%right boundary nodes.
for k=N1_linear+1:N1_linear+N2_linear
    BN_linear_Dirichlet(2,k)=N1_linear*(N2_linear+1)+k-N1_linear;
end

%top boundary nodes.
for k=N1_linear+N2_linear+1:2*N1_linear+N2_linear
    BN_linear_Dirichlet(2,k)=(2*N1_linear+N2_linear+2-k)*(N2_linear+1);
end

%left boundary nodes.
for k=2*N1_linear+N2_linear+1:nbn_linear
    BN_linear_Dirichlet(2,k)=2*N1_linear+2*N2_linear+2-k;
end
%plot boundary nodes for linear element
if plot_mesh_linear==1
    figure('NumberTitle','off','Name',['linear element: boundary nodes with h = ' num2str(h(1))]);
    patch('Faces',T_linear','Vertices',V_linear','FaceColor','w');
    hold on
    V_linear_left=V_linear(:,BN_linear_Dirichlet(2,:));
    plot(V_linear_left(1,:),V_linear_left(2,:),'*');
    text(V_linear_left(1,:),V_linear_left(2,:),num2str(BN_linear_Dirichlet(2,:)'),'FontSize',12,'Color','red');
    axis([min(V_linear(1,:))-0.02 max(V_linear(1,:))+0.02 min(V_linear(2,:))-0.02 max(V_linear(2,:))+0.02]);
    title('Dirichlet boundary nodes for linear element');
    xlabel('x label');
    ylabel('y label');
    hold off
end 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%generate boundary edge information 
%BN_Dirichlet(1:2,k): indices of the two end points of the kth boundary edge among all grid points, not the nodes of FE.
%BN_Dirichlet(3,k): specifiy the type of the kth boundary node.
%BN_Dirichlet(4,k): index of the element which contains the kth boundary edge.
nbe=2*(N1_linear+N2_linear);
BE_linear_Dirichlet=zeros(4,nbe);

%All Dirichlet boundary edges
BE_linear_Dirichlet(3,:)=-1;
%bottom boundary edges.
for k=1:N1_linear
    BE_linear_Dirichlet(4,k)=(k-1)*2*N2_linear+1;
    BE_linear_Dirichlet(1,k)=(k-1)*(N2_linear+1)+1;
    BE_linear_Dirichlet(2,k)=k*(N2_linear+1)+1;
end

%right boundary edges.
for k=N1_linear+1:N1_linear+N2_linear
    BE_linear_Dirichlet(4,k)=(N1_linear-1)*2*N2_linear+2*(k-N1_linear);
    BE_linear_Dirichlet(1,k)=N1_linear*(N2_linear+1)+k-N1_linear;
    BE_linear_Dirichlet(2,k)=N1_linear*(N2_linear+1)+k-N1_linear+1;
end

%top boundary edges.
for k=N1_linear+N2_linear+1:2*N1_linear+N2_linear
    BE_linear_Dirichlet(4,k)=(2*N1_linear+N2_linear+1-k)*2*N2_linear;
    BE_linear_Dirichlet(1,k)=(2*N1_linear+N2_linear+2-k)*(N2_linear+1);
    BE_linear_Dirichlet(2,k)=(2*N1_linear+N2_linear+1-k)*(N2_linear+1);
end

%left boundary edges.
for k=2*N1_linear+N2_linear+1:nbe
    BE_linear_Dirichlet(4,k)=2*(2*N1_linear+2*N2_linear+1-k)-1;
    BE_linear_Dirichlet(1,k)=2*N1_linear+2*N2_linear+2-k;
    BE_linear_Dirichlet(2,k)=2*N1_linear+2*N2_linear+1-k;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%generate mesh information for quadratic element  
   N1_quadratic=(right-left)/h(1);
   N2_quadratic=(top-bottom)/h(2);
   dh=h/2;
   dN1=N1_quadratic*2;
   dN2=N2_quadratic*2;
   tnp_quadratic=(dN1+1)*(dN2+1);
   V_quadratic=zeros(2,tnp_quadratic);
   T_quadratic=zeros(3,2*N1_quadratic*N2_quadratic);
   Q_quadratic=zeros(dN1+1,dN2+1);

   for j=1:tnp_quadratic
      if mod(j,dN2+1)==0
         V_quadratic(1,j)=left+(j/(dN2+1)-1)*dh(1);
         V_quadratic(2,j)=top;
      else
         V_quadratic(1,j)=left+fix(j/(dN2+1))*dh(1);
         V_quadratic(2,j)=bottom+(mod(j,dN2+1)-1)*dh(2);
      end
   end

   for i=1:dN1+1
      for j=1:dN2+1
         Q_quadratic(i,j)=(i-1)*(dN2+1)+j;
      end
   end

%Go through all rectangles in the partition. 
%For the nth rectangle, store the information of its two triangular elements whose element indices are 2n-1 and 2n.
   for n=1:N1_quadratic*N2_quadratic
   
      if mod(n,N2_quadratic)==0
         row=N2_quadratic;
         column=n/N2_quadratic;
      else
         row=mod(n,N2_quadratic);
         column=fix(n/N2_quadratic)+1;
      end
   
      T_quadratic(1,2*n-1)=Q_quadratic(2*column-1,2*row-1);
      T_quadratic(2,2*n-1)=Q_quadratic(2*column+1,2*row-1); 
      T_quadratic(3,2*n-1)=Q_quadratic(2*column-1,2*row+1);
      T_quadratic(4,2*n-1)=Q_quadratic(2*column,2*row-1);
      T_quadratic(5,2*n-1)=Q_quadratic(2*column,2*row);
      T_quadratic(6,2*n-1)=Q_quadratic(2*column-1,2*row);


      T_quadratic(1,2*n)=Q_quadratic(2*column-1,2*row+1);
      T_quadratic(2,2*n)=Q_quadratic(2*column+1,2*row-1);
      T_quadratic(3,2*n)=Q_quadratic(2*column+1,2*row+1);
      T_quadratic(4,2*n)=Q_quadratic(2*column,2*row);
      T_quadratic(5,2*n)=Q_quadratic(2*column+1,2*row);
      T_quadratic(6,2*n)=Q_quadratic(2*column,2*row+1); 
      
   end
   

   %plot mesh for quadratic element
   if plot_mesh_quadratic==1
       figure 
       patch('Faces',T_linear','Vertices',V_linear','FaceColor','w');
       hold on
       plot(V_quadratic(1,:),V_quadratic(2,:),'*');
       text(V_quadratic(1,:),V_quadratic(2,:),num2str((1:size(V_quadratic,2))'),'FontSize',12,'Color','red');
       % order of face displayed in the mean value coordinates of the triangle
       sizeT_linear=size(T_linear);
       T1_linear=zeros(3,size(T_linear,2));
       T2_linear=zeros(3,size(T_linear,2));
       for i=1:sizeT_linear(1)
           for j=1:sizeT_linear(2)
               T1_linear(i,j)=V_linear(1,T_linear(i,j));
               T2_linear(i,j)=V_linear(2,T_linear(i,j));
           end
       end
       Txx_linear=mean(T1_linear);
       Tyy_linear=mean(T2_linear);
       text(Txx_linear(:),Tyy_linear(:),num2str((1:size(T_linear,2))'),'FontSize',10,'HorizontalAlignment','left','BackgroundColor',[.7 .9 .7]);     
       % some parameters
       axis([min(V_quadratic(1,:))-0.02 max(V_quadratic(1,:))+0.02 min(V_quadratic(2,:))-0.02 max(V_quadratic(2,:))+0.02]);
       title('mesh for quadratic element');
       xlabel('x label');
       ylabel('y label');
       hold off
   end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%generate boundary node information for quadratic element
N1_quadratic=N1_quadratic*2;
N2_quadratic=N2_quadratic*2;
nbn_quadratic=2*(N1_quadratic+N2_quadratic);
BN_quadratic_Dirichlet=zeros(2,nbn_quadratic);
%all Dirichlet boundary nodes
BN_quadratic_Dirichlet(1,:)=-1;
%bottom boundary nodes.
for k=1:N1_quadratic
    BN_quadratic_Dirichlet(2,k)=(k-1)*(N2_quadratic+1)+1;
end

%right boundary nodes.
for k=N1_quadratic+1:N1_quadratic+N2_quadratic
    BN_quadratic_Dirichlet(2,k)=N1_quadratic*(N2_quadratic+1)+k-N1_quadratic;
end

%top boundary nodes.
for k=N1_quadratic+N2_quadratic+1:2*N1_quadratic+N2_quadratic
    BN_quadratic_Dirichlet(2,k)=(2*N1_quadratic+N2_quadratic+2-k)*(N2_quadratic+1);
end

%left boundary nodes.
for k=2*N1_quadratic+N2_quadratic+1:nbn_quadratic
    BN_quadratic_Dirichlet(2,k)=2*N1_quadratic+2*N2_quadratic+2-k;
end
%plot boundary nodes for quadratic element
if plot_mesh_quadratic==1
    figure 
    patch('Faces',T_linear','Vertices',V_linear','FaceColor','w');
    hold on
    V_quadratic_left=V_quadratic(:,BN_quadratic_Dirichlet(2,:));
    plot(V_quadratic_left(1,:),V_quadratic_left(2,:),'*');
    text(V_quadratic_left(1,:),V_quadratic_left(2,:),num2str(BN_quadratic_Dirichlet(2,:)'),'FontSize',12,'Color','red');
    axis([min(V_linear(1,:))-0.02 max(V_linear(1,:))+0.02 min(V_linear(2,:))-0.02 max(V_linear(2,:))+0.02]);
    title('Dirichlet boundary for quadratic element');
    xlabel('x label');
    ylabel('y label');
    hold off   
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%generate boundary edge information 
%BN_Dirichlet(1:2,k): indices of the two end points of the kth boundary edge among all grid points, not the nodes of FE.
%BN_Dirichlet(3,k): specifiy the type of the kth boundary node.
%BN_Dirichlet(4,k): index of the element which contains the kth boundary edge.
nbe=2*(N1_quadratic+N2_quadratic);
BE_quadratic_Dirichlet=zeros(4,nbe);

%All Dirichlet boundary edges
BE_quadratic_Dirichlet(3,:)=-1;
%bottom boundary edges.
for k=1:N1_quadratic
    BE_quadratic_Dirichlet(4,k)=(k-1)*2*N2_quadratic+1;
    BE_quadratic_Dirichlet(1,k)=(k-1)*(N2_quadratic+1)+1;
    BE_quadratic_Dirichlet(2,k)=k*(N2_quadratic+1)+1;
end

%right boundary edges.
for k=N1_quadratic+1:N1_quadratic+N2_quadratic
    BE_quadratic_Dirichlet(4,k)=(N1_quadratic-1)*2*N2_quadratic+2*(k-N1_quadratic);
    BE_quadratic_Dirichlet(1,k)=N1_quadratic*(N2_quadratic+1)+k-N1_quadratic;
    BE_quadratic_Dirichlet(2,k)=N1_quadratic*(N2_quadratic+1)+k-N1_quadratic+1;
end

%top boundary edges.
for k=N1_quadratic+N2_quadratic+1:2*N1_quadratic+N2_quadratic
    BE_quadratic_Dirichlet(4,k)=(2*N1_quadratic+N2_quadratic+1-k)*2*N2_quadratic;
    BE_quadratic_Dirichlet(1,k)=(2*N1_quadratic+N2_quadratic+2-k)*(N2_quadratic+1);
    BE_quadratic_Dirichlet(2,k)=(2*N1_quadratic+N2_quadratic+1-k)*(N2_quadratic+1);
end

%left boundary edges.
for k=2*N1_quadratic+N2_quadratic+1:nbe
    BE_quadratic_Dirichlet(4,k)=2*(2*N1_quadratic+2*N2_quadratic+1-k)-1;
    BE_quadratic_Dirichlet(1,k)=2*N1_quadratic+2*N2_quadratic+2-k;
    BE_quadratic_Dirichlet(2,k)=2*N1_quadratic+2*N2_quadratic+1-k;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%