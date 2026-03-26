function [pt3D, erreur] = sintersection( pt1, v1, pt2, v2 )

v1 = snormaliser(v1);
v2 = snormaliser(v2);

n1 = cross(cross(v1,v2),v2);
t1 = dot( n1, (pt2-pt1), 2 )./dot( n1, v1 ,2 );
p1 = pt1 + repmat(t1,1,3).*v1;
p2 = pt2 + repmat( dot( p1-pt2, v2, 2) ,1,3).*v2;

pt3D = (p1 + p2)/2;
erreur = sqrt(sum((p1-p2).^2,2));

function out = snormaliser( in )

out = in ./ repmat(sqrt(dot(in,in,2)),1 , size(in,2) );