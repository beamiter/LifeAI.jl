#!/usr/bin/env julia
# Compose the deployment lockfile from the validated application and the small
# CUDA/NCCL environment, without instantiating unrelated optional artifacts.
using Pkg,TOML

function main(args)
    root=dirname(@__DIR__)
    supplemental=isempty(args) ? "/jfs/application/user/bgl/envs/lifeai-nccl" : abspath(args[1])
    base=TOML.parsefile(joinpath(root,"Manifest.toml"))
    extra=TOML.parsefile(joinpath(supplemental,"Manifest.toml"))
    project=TOML.parsefile(joinpath(root,"Project.toml"))
    for name in ("CUDA","CUDACore","GPUToolbox","CUDA_Runtime_jll","CUDA_Compiler_jll")
        old=only(base["deps"][name]);new=only(extra["deps"][name])
        old["version"]==new["version"] && get(old,"git-tree-sha1",nothing)==get(new,"git-tree-sha1",nothing) ||
            error("Supplemental environment changed validated $name; keep the application versions")
    end
    only(extra["deps"]["NCCL"])["version"]=="0.1.3" || error("Expected the validated NCCL 0.1.3 API")
    for (name,entries) in extra["deps"]
        base["deps"][name]=entries
    end
    base["deps"]["LifeAI"]=[Dict("deps"=>sort!(collect(keys(project["deps"]))),"path"=>"../../..",
        "uuid"=>project["uuid"],"version"=>project["version"],"weakdeps"=>project["weakdeps"],"extensions"=>project["extensions"])]
    names=("LifeAI","CUDA","NCCL","NCCL_jll","CUDA_Driver_jll","CUDA_Runtime_jll","CUDA_Compiler_jll","HTTP","JSON3","BFloat16s")
    deps=Dict(name=>only(base["deps"][name])["uuid"] for name in names)
    config=Dict("deps"=>deps,"compat"=>Dict("julia"=>"1.12","CUDA"=>"=6.2.1","NCCL"=>"=0.1.3","NCCL_jll"=>"=2.28.3"),
        "preferences"=>Dict("CUDA_Runtime_jll"=>Dict("version"=>"12.9"),"CUDA_Compiler_jll"=>Dict("version"=>"12.9")))
    target=joinpath(root,"configs","deployment","ripple_tp8");mkpath(target)
    file=joinpath(target,"Project.toml")
    open(io->TOML.print(io,config;sorted=true),file,"w")
    if isdefined(Pkg.Types,:project_resolve_hash)
        base["project_hash"]=Pkg.Types.project_resolve_hash(Pkg.Types.read_project(file))
    else
        pop!(base,"project_hash",nothing)
    end
    open(io->TOML.print(io,base;sorted=true),joinpath(target,"Manifest.toml"),"w")
    println("Deployment environment prepared with a relative immutable LifeAI source and CUDA 12.9 preferences")
end
main(ARGS)
