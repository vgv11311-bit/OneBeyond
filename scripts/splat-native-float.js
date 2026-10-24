// Full-precision Gaussian rendering for explicit review candidates. No image
// projection, geometry synthesis, sharpening, packing or automatic fallback.
import * as T from 'three';

export async function createNativeFloatRenderer({renderer,url,transform,onChange=()=>{},onError=()=>{}}){
 if(!renderer.extensions.has('EXT_color_buffer_float'))throw Error('Full-precision review requires floating-point render targets.');
 const response=await fetch(url);if(!response.ok)throw Error('Could not load the Gaussian asset.');
 const bytes=await response.arrayBuffer(),raw=new Uint8Array(bytes);let end=0;
 for(let i=0;i<Math.min(raw.length,65536);i++)if(raw[i]===10&&new TextDecoder().decode(raw.subarray(Math.max(0,i-10),i)).endsWith('end_header')){end=i+1;break;}
 if(!end)throw Error('Invalid Gaussian PLY header.');
 const header=new TextDecoder().decode(raw.subarray(0,end));
 if(!header.includes('format binary_little_endian 1.0'))throw Error('Full-precision review requires binary little-endian PLY.');
 const declarations=header.split(/\r?\n/).filter(line=>line.startsWith('property '));
 if(declarations.some(line=>!/^property float \w+$/.test(line)))throw Error('Unsupported Gaussian PLY property type.');
 const names=declarations.map(line=>line.split(' ')[2]),count=Number(header.match(/element vertex (\d+)/)?.[1]);
 const indices=Object.fromEntries(names.map((n,i)=>[n,i]));
 const required=['x','y','z','opacity',...Array.from({length:3},(_,i)=>'scale_'+i),...Array.from({length:4},(_,i)=>'rot_'+i),...Array.from({length:3},(_,i)=>'f_dc_'+i),...Array.from({length:45},(_,i)=>'f_rest_'+i)];
 if(!Number.isInteger(count)||count<1||count>2000000||required.some(n=>indices[n]===undefined)||raw.length-end!==count*names.length*4)throw Error('Invalid Gaussian PLY layout or size.');
 const floats=new Float32Array(bytes.slice(end)),get=(i,n)=>floats[i*names.length+indices[n]];
 const texels=20,W=Math.min(2048,renderer.capabilities.maxTextureSize),H=Math.ceil(count*texels/W);
 if(H>renderer.capabilities.maxTextureSize)throw Error('Gaussian asset exceeds this device texture limit.');
 const data=new Float32Array(W*H*4),positions=new Float32Array(count*3),matrix=new T.Matrix4(),quat=new T.Quaternion();
 for(let i=0;i<count;i++){
  const p=i*texels*4;
  for(let j=0;j<3;j++)data[p+j]=positions[i*3+j]=get(i,['x','y','z'][j]);
  data[p+3]=1/(1+Math.exp(-get(i,'opacity')));
  quat.set(get(i,'rot_1'),get(i,'rot_2'),get(i,'rot_3'),get(i,'rot_0'));
  if(quat.lengthSq()<1e-12)throw Error('Invalid Gaussian rotation.');quat.normalize();matrix.makeRotationFromQuaternion(quat);
  const m=matrix.elements,s=[0,1,2].map(j=>Math.exp(2*get(i,'scale_'+j))),cov=(r,c)=>s.reduce((v,x,j)=>v+m[j*4+r]*m[j*4+c]*x,0);
  data[p+4]=cov(0,0);data[p+5]=cov(1,1);data[p+6]=cov(2,2);data[p+8]=cov(0,1);data[p+9]=cov(0,2);data[p+10]=cov(1,2);
  for(let degree=0;degree<16;degree++)for(let channel=0;channel<3;channel++)data[p+12+degree*4+channel]=get(i,degree===0?'f_dc_'+channel:'f_rest_'+(channel*15+degree-1));
 }
 if(data.some(value=>!Number.isFinite(value)))throw Error('Non-finite Gaussian data.');
 const texture=new T.DataTexture(data,W,H,T.RGBAFormat,T.FloatType);texture.needsUpdate=true;
 const target=new T.WebGLRenderTarget(1,1,{type:T.HalfFloatType,depthBuffer:true});
 const geometry=new T.InstancedBufferGeometry();geometry.setAttribute('position',new T.Float32BufferAttribute([-1,-1,0,1,-1,0,1,1,0,-1,1,0],3));geometry.setIndex([0,1,2,0,2,3]);geometry.instanceCount=count;
 const order=new T.InstancedBufferAttribute(new Float32Array(count),1);geometry.setAttribute('aIndex',order);
 const model=new T.Matrix4();if(transform)model.set(...transform.flat());
 const inverseModel=model.clone().invert(),flip=new T.Matrix4().makeScale(1,-1,-1),bufferSize=new T.Vector2();
 const uniforms={splats:{value:texture},viewCV:{value:new T.Matrix4()},projection:{value:new T.Matrix4()},origin:{value:new T.Vector3()},K:{value:new T.Vector4()},size:{value:new T.Vector2()},width:{value:W},nearFar:{value:new T.Vector2(.1,1000)}};
 const material=new T.ShaderMaterial({glslVersion:T.GLSL3,side:T.DoubleSide,transparent:true,depthTest:true,depthWrite:false,blending:T.CustomBlending,blendSrc:T.OneFactor,blendDst:T.OneMinusSrcAlphaFactor,uniforms,
 vertexShader:`in float aIndex;uniform sampler2D splats;uniform mat4 viewCV,projection;uniform vec3 origin;uniform vec4 K;uniform vec2 size,nearFar;uniform int width;out vec2 uvG;flat out vec3 colour;flat out float alpha;
 vec4 readP(int o){int a=int(aIndex)*20+o;return texelFetch(splats,ivec2(a%width,a/width),0);}
 void main(){vec4 p=readP(0),d=readP(1),o=readP(2);vec3 v=(viewCV*vec4(p.xyz,1)).xyz;uvG=position.xy*3.;alpha=0.;colour=vec3(0.);
 if(v.z<nearFar.x||v.z>nearFar.y){gl_Position=vec4(2,2,2,1);return;}
 vec3 direction=normalize(p.xyz-origin);float x=direction.x,y=direction.y,z=direction.z,xx=x*x,yy=y*y,zz=z*z;
 float sh[16];sh[0]=.28209479177387814;sh[1]=-.4886025119029199*y;sh[2]=.4886025119029199*z;sh[3]=-.4886025119029199*x;
 sh[4]=1.0925484305920792*x*y;sh[5]=-1.0925484305920792*y*z;sh[6]=.31539156525252005*(2.*zz-xx-yy);sh[7]=-1.0925484305920792*x*z;sh[8]=.5462742152960396*(xx-yy);
 sh[9]=-.5900435899266435*y*(3.*xx-yy);sh[10]=2.890611442640554*x*y*z;sh[11]=-.4570457994644658*y*(4.*zz-xx-yy);sh[12]=.3731763325901154*z*(2.*zz-3.*xx-3.*yy);sh[13]=-.4570457994644658*x*(4.*zz-xx-yy);sh[14]=1.445305721320277*z*(xx-yy);sh[15]=-.5900435899266435*x*(xx-3.*yy);
 colour=vec3(.5);for(int i=0;i<16;i++)colour+=readP(3+i).rgb*sh[i];colour=max(vec3(0),colour);
 mat3 covariance=mat3(d.x,o.x,o.y,o.x,d.y,o.z,o.y,o.z,d.z),R=mat3(viewCV);covariance=R*covariance*transpose(R);
 float tx=clamp(v.x/v.z,-1.3*size.x/(2.*K.x),1.3*size.x/(2.*K.x)),ty=clamp(v.y/v.z,-1.3*size.y/(2.*K.y),1.3*size.y/(2.*K.y));
 vec3 jx=vec3(K.x/v.z,0.,-K.x*tx/v.z),jy=vec3(0.,K.y/v.z,-K.y*ty/v.z);
 float a=dot(jx,covariance*jx),b=dot(jx,covariance*jy),c=dot(jy,covariance*jy),det0=max(0.,a*c-b*b);a+=.1;c+=.1;float det=max(1e-10,a*c-b*b);
 alpha=p.w*sqrt(det0/det);float mid=.5*(a+c),disc=sqrt(max(0.,.25*(a-c)*(a-c)+b*b)),l1=max(.0001,mid+disc),l2=max(.0001,mid-disc);
 vec2 e1=abs(b)>1e-6?normalize(vec2(b,l1-a)):(a>=c?vec2(1,0):vec2(0,1)),e2=vec2(-e1.y,e1.x);
 vec2 pixel=K.xy*v.xy/v.z+K.zw+e1*sqrt(l1)*uvG.x+e2*sqrt(l2)*uvG.y;
 vec4 clip=projection*vec4(v.x,-v.y,-v.z,1.);gl_Position=vec4((vec2(pixel.x/size.x*2.-1.,1.-pixel.y/size.y*2.))*clip.w,clip.z,clip.w);}`,
 fragmentShader:`in vec2 uvG;flat in vec3 colour;flat in float alpha;out vec4 fragment;void main(){float power=dot(uvG,uvG);if(power>9.)discard;float a=min(.999,alpha*exp(-.5*power));if(a<1./255.)discard;fragment=vec4(colour*a,a);}`});
 const scene=new T.Scene(),mesh=new T.Mesh(geometry,material);mesh.frustumCulled=false;scene.add(mesh);
 const display=new T.Scene(),displayMaterial=new T.ShaderMaterial({depthTest:false,depthWrite:false,uniforms:{image:{value:target.texture}},vertexShader:'varying vec2 uvP;void main(){uvP=uv;gl_Position=vec4(position.xy,0,1);}',fragmentShader:'varying vec2 uvP;uniform sampler2D image;void main(){gl_FragColor=vec4(clamp(texture2D(image,uvP).rgb,0.,1.),1.);}'}),quad=new T.Mesh(new T.PlaneGeometry(2,2),displayMaterial),dummyCamera=new T.Camera();display.add(quad);
 // Rotation changes depth ordering; translation adds the same depth offset to
 // every centre. Sort asynchronously, without blocking mouse/WASD rendering.
 const workerCode=`let positions,order,temp,keys;const f=new Float32Array(1),u=new Uint32Array(f.buffer);onmessage=e=>{if(e.data.positions){positions=e.data.positions;const n=positions.length/3;order=new Uint32Array(n);temp=new Uint32Array(n);keys=new Uint32Array(n);return;}const v=e.data.view,n=order.length;for(let i=0;i<n;i++){f[0]=v[2]*positions[i*3]+v[6]*positions[i*3+1]+v[10]*positions[i*3+2];const k=u[0];keys[i]=((k&0x80000000)?~k:(k^0x80000000))>>>0;order[i]=i;}for(let shift=0;shift<32;shift+=8){const bins=new Uint32Array(256);for(let i=0;i<n;i++)bins[(keys[order[i]]>>>shift)&255]++;let sum=0;for(let j=0;j<256;j++){const x=bins[j];bins[j]=sum;sum+=x;}for(let i=0;i<n;i++){const index=order[i];temp[bins[(keys[index]>>>shift)&255]++]=index;}const swap=order;order=temp;temp=swap;}const result=new Float32Array(n);for(let i=0;i<n;i++)result[i]=order[n-i-1];postMessage({order:result,view:v},[result.buffer]);};`;
 const workerUrl=URL.createObjectURL(new Blob([workerCode],{type:'text/javascript'})),worker=new Worker(workerUrl);worker.postMessage({positions},[positions.buffer]);
 let busy=false,disposed=false,lastRotation=null,latestView=null,sorts=0,sortError=null;
 const sameRotation=(a,b)=>a&&[2,6,10].every(i=>Math.abs(a[i]-b[i])<1e-7);
 const requestSort=()=>{if(disposed||busy||!latestView||sameRotation(lastRotation,latestView))return;busy=true;worker.postMessage({view:latestView});};
 worker.onmessage=e=>{if(disposed)return;order.array.set(e.data.order);order.needsUpdate=true;lastRotation=e.data.view;busy=false;sorts++;onChange();requestSort();};
 worker.onerror=()=>{if(!disposed){busy=false;sortError='Gaussian depth sorter failed.';worker.terminate();onError(Error(sortError));}};
 function update(camera){
  if(sortError)throw Error(sortError);
  camera.updateMatrixWorld();camera.matrixWorldInverse.copy(camera.matrixWorld).invert();uniforms.viewCV.value.copy(flip).multiply(camera.matrixWorldInverse).multiply(model);
  uniforms.projection.value.copy(camera.projectionMatrix);uniforms.origin.value.setFromMatrixPosition(camera.matrixWorld).applyMatrix4(inverseModel);
  renderer.getDrawingBufferSize(bufferSize);const w=bufferSize.x,h=bufferSize.y;if(target.width!==w||target.height!==h)target.setSize(w,h);
  uniforms.size.value.set(w,h);const p=camera.projectionMatrix.elements;uniforms.K.value.set(p[0]*w/2,p[5]*h/2,(1-p[8])*w/2,(1+p[9])*h/2);uniforms.nearFar.value.set(camera.near,camera.far);
  latestView=uniforms.viewCV.value.toArray();requestSort();
 }
 return {
  update,
  render(camera,drawBase){
   update(camera);if(!sorts)return;
   const previousTarget=renderer.getRenderTarget(),colour=renderer.getClearColor(new T.Color()),alpha=renderer.getClearAlpha();
   renderer.setRenderTarget(target);renderer.clear();drawBase?.();renderer.render(scene,dummyCamera);
   renderer.setRenderTarget(previousTarget);renderer.render(display,dummyCamera);renderer.setClearColor(colour,alpha);
  },
  inspect:()=>({renderer:'native-float-v1',count,sorts,sortPending:busy,buffer:[target.width,target.height],source:'float32 covariance and degree-3 SH',accumulation:'RGBA16F'}),
  dispose(){disposed=true;worker.terminate();URL.revokeObjectURL(workerUrl);texture.dispose();target.dispose();geometry.dispose();material.dispose();quad.geometry.dispose();displayMaterial.dispose();},
 };
}
