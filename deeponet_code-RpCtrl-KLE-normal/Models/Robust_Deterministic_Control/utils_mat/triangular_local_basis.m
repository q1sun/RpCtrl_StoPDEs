function r=triangular_local_basis(x,y,vertices,basis_type,basis_index,derivative_degree_x,derivative_degree_y)
%This is for the local basis functions of triangular FE.
%x,y: the coordinates of the point where we want to evaluate the local FE basis function.
%left_lower_point: The coordinates of the left-lower vertice of the current element whose FE local basis we are evaluating. 
%h_partition: the step size of the partition.
%basis_type: the type of the FE.
%basis_type=1:2D linear FE.
%basis_type=2:2D Lagrange quadratic FE.
%basis_index: the index of basis function to specify which basis function we want to use.
%derivative_degree_x:the derivative degree of the FE basis function with respect to x.
%derivative_degree_y:the derivative degree of the FE basis function with respect to y.

%J is the Jacobi matrix of the affine mapping, which has four elememts J_11, J_12, J_21, J_22.
%J_det is the determinant of J.
%x_hat,y_hat: the '\hat{x}' and '\hat{y}' in the affine mapping, which are for the coordinates of
%the reference element

J_11=vertices(1,2)-vertices(1,1);% x2-x1
J_12=vertices(1,3)-vertices(1,1);% x3-x1
J_21=vertices(2,2)-vertices(2,1);% y2-y1
J_22=vertices(2,3)-vertices(2,1);% y3-y1
J_det=J_11*J_22-J_12*J_21;

x_hat=(J_22*(x-vertices(1,1))-J_12*(y-vertices(2,1)))/J_det;
y_hat=(-J_21*(x-vertices(1,1))+J_11*(y-vertices(2,1)))/J_det;

if derivative_degree_x==0&&derivative_degree_y==0
    r=triangular_reference_basis(x_hat,y_hat,basis_type,basis_index,0,0);
elseif derivative_degree_x==1&&derivative_degree_y==0
    r=(triangular_reference_basis(x_hat,y_hat,basis_type,basis_index,1,0)*J_22+triangular_reference_basis(x_hat,y_hat,basis_type,basis_index,0,1)*(-J_21))/J_det;
elseif derivative_degree_x==0&&derivative_degree_y==1
    r=(triangular_reference_basis(x_hat,y_hat,basis_type,basis_index,1,0)*(-J_12)+triangular_reference_basis(x_hat,y_hat,basis_type,basis_index,0,1)*J_11)/J_det;
elseif derivative_degree_x==2&&derivative_degree_y==0
    r=(triangular_reference_basis(x_hat,y_hat,basis_type,basis_index,2,0)*J_22^2+triangular_reference_basis(x_hat,y_hat,basis_type,basis_index,0,2)*J_21^2+triangular_reference_basis(x_hat,y_hat,basis_type,basis_index,1,1)*(-2*J_21*J_22))/J_det^2;
elseif derivative_degree_x==0&&derivative_degree_y==2
    r=(triangular_reference_basis(x_hat,y_hat,basis_type,basis_index,2,0)*J_12^2+triangular_reference_basis(x_hat,y_hat,basis_type,basis_index,0,2)*J_11^2+triangular_reference_basis(x_hat,y_hat,basis_type,basis_index,1,1)*(-2*J_11*J_12))/J_det^2;
elseif derivative_degree_x==1&&derivative_degree_y==1
    r=(triangular_reference_basis(x_hat,y_hat,basis_type,basis_index,2,0)*(-J_22*J_12)+triangular_reference_basis(x_hat,y_hat,basis_type,basis_index,0,2)*(-J_21*J_11)+triangular_reference_basis(x_hat,y_hat,basis_type,basis_index,1,1)*(J_21*J_12+J_11*J_22))/J_det^2;
end











function r=triangular_reference_basis(x,y,basis_type,basis_index,derivative_degree_x,derivative_degree_y)
%This is the reference FE basis function on triangle ABC where A=(0,0), B=(1,0) and C=(0,1).
%x,y: the coordinates of the point where we want to evaluate the reference FE basis function.
%basis_type: the type of the FE.
%basis_type=1:2D linear FE.
%basis_type=2:2D Lagrange quadratic FE.
%basis_index: the index of FE basis function to specify which FE basis function we want to use.
%derivative_degree_x:the derivative degree of the FE basis function with respect to x.
%derivative_degree_y:the derivative degree of the FE basis function with respect to y.


if basis_type==2
    
    if derivative_degree_x==0&&derivative_degree_y==0
        
        if basis_index==1
            r=1-3*x-3*y+2*x.^2+2*y.^2+4*x.*y;
        elseif basis_index==2
            r=2*x.^2-x;
        elseif basis_index==3
            r=2*y.^2-y;
        elseif basis_index==4
            r=4*x-4*x.^2-4*x.*y;
        elseif basis_index==5
            r=4*x.*y;
        elseif basis_index==6
            r=4*y-4*y.^2-4*x.*y;
        end
             
    elseif derivative_degree_x==1&&derivative_degree_y==0
 
        if basis_index==1
            r=-3+4*x+4*y;
        elseif basis_index==2
            r=4*x-1;
        elseif basis_index==3
            r=0;
        elseif basis_index==4
            r=4-8*x-4*y;
        elseif basis_index==5
            r=4*y;
        elseif basis_index==6
            r=-4*y;
        end           

                      
    elseif derivative_degree_x==0&&derivative_degree_y==1
            
        if basis_index==1
            r=-3+4*y+4*x;
        elseif basis_index==2
            r=0;
        elseif basis_index==3
            r=4*y-1;
        elseif basis_index==4
            r=-4*x;
        elseif basis_index==5
            r=4*x;
        elseif basis_index==6
            r=4-8*y-4*x;
        end
      
    elseif derivative_degree_x==2&&derivative_degree_y==0  
        
        if basis_index==1
            r=4;
        elseif basis_index==2
            r=4;
        elseif basis_index==3
            r=0;
        elseif basis_index==4
            r=-8;
        elseif basis_index==5
            r=0;
        elseif basis_index==6
            r=0;
        end

    elseif derivative_degree_x==0&&derivative_degree_y==2 

        if basis_index==1
            r=4;
        elseif basis_index==2
            r=0;
        elseif basis_index==3
            r=4;
        elseif basis_index==4
            r=0;
        elseif basis_index==5
            r=0;
        elseif basis_index==6
            r=-8;
        end

    elseif derivative_degree_x==1&&derivative_degree_y==1 

        if basis_index==1
            r=4;
        elseif basis_index==2
            r=0;
        elseif basis_index==3
            r=0;
        elseif basis_index==4
            r=-4;
        elseif basis_index==5
            r=4;
        elseif basis_index==6
            r=-4;
        end 
      
    end


elseif basis_type==1

    if derivative_degree_x==0&&derivative_degree_y==0
        
        if basis_index==1
            r=1-x-y;
        elseif basis_index==2
            r=x;
        elseif basis_index==3
            r=y;
        end

    elseif derivative_degree_x==1&&derivative_degree_y==0
        
        if basis_index==1
            r=-1;
        elseif basis_index==2
            r=1;
        elseif basis_index==3
            r=0;
        end

    elseif derivative_degree_x==0&&derivative_degree_y==1
        
        if basis_index==1
            r=-1;
        elseif basis_index==2
            r=0;
        elseif basis_index==3
            r=1;
        end
        
    end
       
end