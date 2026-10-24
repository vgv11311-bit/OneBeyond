// Pure numeric, scale-aware swept collision. No room-specific object positions.
export const WALK_SETTINGS=Object.freeze({speed:4,look:.0022,damping:15,sprint:1.8,referenceEyeHeight:1.4});
export function photographicWalkSpeed(eyeHeight){
 return Number.isFinite(eyeHeight)&&eyeHeight>0?WALK_SETTINGS.speed*eyeHeight/WALK_SETTINGS.referenceEyeHeight:WALK_SETTINGS.speed;
}
export function createNavigation(data) {
 if (!data) return null;
 const {bounds, radius, cell, width, depth} = data;
 const finiteVector=(v,n)=>Array.isArray(v)&&v.length===n&&v.every(Number.isFinite);
 if (!Number.isFinite(radius)||!Number.isFinite(cell)||!(radius>0&&cell>0)||!Number.isInteger(width)||!Number.isInteger(depth)||width<1||depth<1||width*depth>4000000||
     !finiteVector(bounds?.min,3)||!finiteVector(bounds?.max,3)||bounds.min.some((v,i)=>v>=bounds.max[i])||
     [0,2].some(i=>bounds.max[i]-bounds.min[i]<=radius*2)||!Array.isArray(data.occupied)||data.occupied.some(i=>!Number.isInteger(i)||i<0||i>=width*depth)) throw new Error('Invalid navigation geometry');
 if(data.enclosure_rect){
  const r=data.enclosure_rect;
  if(!finiteVector(r.center,2)||!finiteVector(r.min,2)||!finiteVector(r.max,2)||!Number.isFinite(r.yaw)||r.min.some((v,i)=>r.max[i]-v<=radius*2))throw new Error('Invalid rectangular enclosure');
 }
 const occupied=new Set(data.occupied);
 const blocked=p=>{
  if(!finiteVector(p,3)||[0,2].some(i=>p[i]<bounds.min[i]+radius||p[i]>bounds.max[i]-radius))return true;
  if(data.enclosure_rect){
   const rect=data.enclosure_rect,c=Math.cos(rect.yaw),s=Math.sin(rect.yaw),dx=p[0]-rect.center[0],dz=p[2]-rect.center[1];
   const local=[c*dx-s*dz,s*dx+c*dz];
   if(local.some((v,i)=>v<rect.min[i]+radius||v>rect.max[i]-radius))return true;
  }
  const x=(p[0]-bounds.min[0])/cell,z=(p[2]-bounds.min[2])/cell,r=radius/cell;
  for(let j=Math.floor(z-r);j<=Math.floor(z+r);j++)for(let i=Math.floor(x-r);i<=Math.floor(x+r);i++){
   if(i<0||j<0||i>=width||j>=depth)continue;
   const dx=Math.max(i-x,0,x-i-1),dz=Math.max(j-z,0,z-j-1);
   if(dx*dx+dz*dz<r*r&&occupied.has(j*width+i))return true;
  }
  return false;
 };
 function constrain(from,to){
  if(!finiteVector(from,3))throw new Error('Invalid walking origin');
  if(!finiteVector(to,3))return [...from];
  const result=[...from], target=[...to];
  // Walking, not flying: keep the capture's eye height. Camera selection may
  // still move to a different registered camera elevation.
  target[1]=from[1];
  for(const axis of [0,2])target[axis]=Math.max(bounds.min[axis]+radius,Math.min(bounds.max[axis]-radius,target[axis]));
  const distance=Math.hypot(target[0]-from[0],target[2]-from[2]);
  const steps=Math.ceil(distance/Math.min(radius*.4,cell*.5))||1;
  // Reject unreasonable teleports rather than allowing tunnelling or hanging.
  if(steps>10000)return result;
  const delta=[(target[0]-from[0])/steps,0,(target[2]-from[2])/steps];
  for(let step=0;step<steps;step++)for(const axis of [0,2]){
   const candidate=[...result];candidate[axis]=Math.max(bounds.min[axis]+radius,Math.min(bounds.max[axis]-radius,result[axis]+delta[axis]));
   if(!blocked(candidate))result[axis]=candidate[axis];
  }
  return result;
 }
 return {constrain,blocked,inspect:()=>({bounds,radius,occupiedCells:occupied.size,source:data.source||'estimated multi-view depth'})};
}

export function selectSurface(position,forward,cameras,previous=-1){
 const scores=cameras.map(c=>{
  const distance=Math.hypot(...position.map((v,i)=>v-c.position[i]));
  const direction=forward.reduce((sum,v,i)=>sum-v*c.rotation[i][2],0);
  return distance+Math.max(0,1-direction)*(c.supportScale||1)*2;
 });
 let best=scores.indexOf(Math.min(...scores));
 // Hysteresis avoids flickering between adjacent source photos while walking.
 if(previous>=0&&scores[previous]<=scores[best]*1.12+1e-6)best=previous;
 return best;
}
