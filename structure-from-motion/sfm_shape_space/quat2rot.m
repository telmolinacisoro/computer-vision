%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%              LABORATORY #5 
%%%              COMPUTER VISION 2023-2024
%%%              NON-RIGID STRUCTURE FROM MOTION - OPTIMIZATION 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function r = quat2rot(q)

q=q/norm(q);
r=zeros(3);
r(1,1) = 2*(q(1)^2 + q(2)^2) - 1;
r(1,2) = 2*(q(2)*q(3) - q(1)*q(4));
r(1,3) = 2*(q(2)*q(4) +q(1)*q(3));

r(2,1) = 2*(q(2)*q(3) + q(1)*q(4));
r(2,2) = 2*(q(1)^2 + q(3)^2) - 1;
r(2,3) = 2*(q(3)*q(4) - q(1)*q(2));

r(3,1) = 2*(q(2)*q(4) - q(1)*q(3));
r(3,2) = 2*(q(3)*q(4) + q(1)*q(2));
r(3,3) = 2*(q(1)^2 + q(4)^2) - 1;

end

function dR = Jacobian_quaternion(q,i)
a = q(1); b = q(2); c = q(3); d = q(4);
if i==1
    dR =matrix([[(4*a)/(a^2 + b^2 + c^2 + d^2) - (4*a^3)/(a^2 + b^2 + c^2 + d^2)^2 - (4*a*b^2)/(a^2 + b^2 + c^2 + d^2)^2,(4*a^2*d)/(a^2 + b^2 + c^2 + d^2)^2 - (2*d)/(a^2 + b^2 + c^2 + d^2) - (4*a*b*c)/(a^2 + b^2 + c^2 + d^2)^2,(2*c)/(a^2 + b^2 + c^2 + d^2) - (4*a^2*c)/(a^2 + b^2 + c^2 + d^2)^2 - (4*a*b*d)/(a^2 + b^2 + c^2 + d^2)^2],[(2*d)/(a^2 + b^2 + c^2 + d^2) - (4*a^2*d)/(a^2 + b^2 + c^2 + d^2)^2 - (4*a*b*c)/(a^2 + b^2 + c^2 + d^2)^2,(4*a)/(a^2 + b^2 + c^2 + d^2) - (4*a^3)/(a^2 + b^2 + c^2 + d^2)^2 - (4*a*c^2)/(a^2 + b^2 + c^2 + d^2)^2,(4*a^2*b)/(a^2 + b^2 + c^2 + d^2)^2 - (2*b)/(a^2 + b^2 + c^2 + d^2) - (4*a*c*d)/(a^2 + b^2 + c^2 + d^2)^2],[(4*a^2*c)/(a^2 + b^2 + c^2 + d^2)^2 - (2*c)/(a^2 + b^2 + c^2 + d^2) - (4*a*b*d)/(a^2 + b^2 + c^2 + d^2)^2,(2*b)/(a^2 + b^2 + c^2 + d^2) - (4*a^2*b)/(a^2 + b^2 + c^2 + d^2)^2 - (4*a*c*d)/(a^2 + b^2 + c^2 + d^2)^2,(4*a)/(a^2 + b^2 + c^2 + d^2) - (4*a^3)/(a^2 + b^2 + c^2 + d^2)^2 - (4*a*d^2)/(a^2 + b^2 + c^2 + d^2)^2]]);
end
if i == 2
    dR = matrix([[(4*b)/(a^2 + b^2 + c^2 + d^2) - (4*b^3)/(a^2 + b^2 + c^2 + d^2)^2 - (4*a^2*b)/(a^2 + b^2 + c^2 + d^2)^2,(2*c)/(a^2 + b^2 + c^2 + d^2) - (4*b^2*c)/(a^2 + b^2 + c^2 + d^2)^2 + (4*a*b*d)/(a^2 + b^2 + c^2 + d^2)^2,(2*d)/(a^2 + b^2 + c^2 + d^2) - (4*b^2*d)/(a^2 + b^2 + c^2 + d^2)^2 - (4*a*b*c)/(a^2 + b^2 + c^2 + d^2)^2],[(2*c)/(a^2 + b^2 + c^2 + d^2) - (4*b^2*c)/(a^2 + b^2 + c^2 + d^2)^2 - (4*a*b*d)/(a^2 + b^2 + c^2 + d^2)^2,- (4*a^2*b)/(a^2 + b^2 + c^2 + d^2)^2 - (4*b*c^2)/(a^2 + b^2 + c^2 + d^2)^2,(4*a*b^2)/(a^2 + b^2 + c^2 + d^2)^2 - (2*a)/(a^2 + b^2 + c^2 + d^2) - (4*b*c*d)/(a^2 + b^2 + c^2 + d^2)^2],[(2*d)/(a^2 + b^2 + c^2 + d^2) - (4*b^2*d)/(a^2 + b^2 + c^2 + d^2)^2 + (4*a*b*c)/(a^2 + b^2 + c^2 + d^2)^2,(2*a)/(a^2 + b^2 + c^2 + d^2) - (4*a*b^2)/(a^2 + b^2 + c^2 + d^2)^2 - (4*b*c*d)/(a^2 + b^2 + c^2 + d^2)^2,- (4*a^2*b)/(a^2 + b^2 + c^2 + d^2)^2 - (4*b*d^2)/(a^2 + b^2 + c^2 + d^2)^2]]);
end
if i == 3
    dR = matrix([[- (4*a^2*c)/(a^2 + b^2 + c^2 + d^2)^2 - (4*b^2*c)/(a^2 + b^2 + c^2 + d^2)^2,(2*b)/(a^2 + b^2 + c^2 + d^2) - (4*b*c^2)/(a^2 + b^2 + c^2 + d^2)^2 + (4*a*c*d)/(a^2 + b^2 + c^2 + d^2)^2,(2*a)/(a^2 + b^2 + c^2 + d^2) - (4*a*c^2)/(a^2 + b^2 + c^2 + d^2)^2 - (4*b*c*d)/(a^2 + b^2 + c^2 + d^2)^2],[(2*b)/(a^2 + b^2 + c^2 + d^2) - (4*b*c^2)/(a^2 + b^2 + c^2 + d^2)^2 - (4*a*c*d)/(a^2 + b^2 + c^2 + d^2)^2,(4*c)/(a^2 + b^2 + c^2 + d^2) - (4*c^3)/(a^2 + b^2 + c^2 + d^2)^2 - (4*a^2*c)/(a^2 + b^2 + c^2 + d^2)^2,(2*d)/(a^2 + b^2 + c^2 + d^2) - (4*c^2*d)/(a^2 + b^2 + c^2 + d^2)^2 + (4*a*b*c)/(a^2 + b^2 + c^2 + d^2)^2],[(4*a*c^2)/(a^2 + b^2 + c^2 + d^2)^2 - (2*a)/(a^2 + b^2 + c^2 + d^2) - (4*b*c*d)/(a^2 + b^2 + c^2 + d^2)^2,(2*d)/(a^2 + b^2 + c^2 + d^2) - (4*c^2*d)/(a^2 + b^2 + c^2 + d^2)^2 - (4*a*b*c)/(a^2 + b^2 + c^2 + d^2)^2,- (4*a^2*c)/(a^2 + b^2 + c^2 + d^2)^2 - (4*c*d^2)/(a^2 + b^2 + c^2 + d^2)^2]]);
end
if i == 4
    dR = matrix([[- (4*a^2*d)/(a^2 + b^2 + c^2 + d^2)^2 - (4*b^2*d)/(a^2 + b^2 + c^2 + d^2)^2,(4*a*d^2)/(a^2 + b^2 + c^2 + d^2)^2 - (2*a)/(a^2 + b^2 + c^2 + d^2) - (4*b*c*d)/(a^2 + b^2 + c^2 + d^2)^2,(2*b)/(a^2 + b^2 + c^2 + d^2) - (4*b*d^2)/(a^2 + b^2 + c^2 + d^2)^2 - (4*a*c*d)/(a^2 + b^2 + c^2 + d^2)^2],[(2*a)/(a^2 + b^2 + c^2 + d^2) - (4*a*d^2)/(a^2 + b^2 + c^2 + d^2)^2 - (4*b*c*d)/(a^2 + b^2 + c^2 + d^2)^2,- (4*a^2*d)/(a^2 + b^2 + c^2 + d^2)^2 - (4*c^2*d)/(a^2 + b^2 + c^2 + d^2)^2,(2*c)/(a^2 + b^2 + c^2 + d^2) - (4*c*d^2)/(a^2 + b^2 + c^2 + d^2)^2 + (4*a*b*d)/(a^2 + b^2 + c^2 + d^2)^2],[(2*b)/(a^2 + b^2 + c^2 + d^2) - (4*b*d^2)/(a^2 + b^2 + c^2 + d^2)^2 + (4*a*c*d)/(a^2 + b^2 + c^2 + d^2)^2,(2*c)/(a^2 + b^2 + c^2 + d^2) - (4*c*d^2)/(a^2 + b^2 + c^2 + d^2)^2 - (4*a*b*d)/(a^2 + b^2 + c^2 + d^2)^2,(4*d)/(a^2 + b^2 + c^2 + d^2) - (4*d^3)/(a^2 + b^2 + c^2 + d^2)^2 - (4*a^2*d)/(a^2 + b^2 + c^2 + d^2)^2]]);
end

end
