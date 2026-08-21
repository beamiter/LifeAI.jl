using Test
using LifeAI: qwen3_vl_checkpoint_spec, verify_qwen3_vl_checkpoint

@testset "Qwen3-VL optional real 2B checkpoint verification" begin
    model_dir = strip(get(ENV, "LIFEAI_QWEN3_VL_2B_MODEL_DIR", ""))
    if isempty(model_dir)
        @info "Skipping real Qwen3-VL-2B checkpoint verification; LIFEAI_QWEN3_VL_2B_MODEL_DIR is unset (no download is attempted)"
        @test_skip false
    else
        @test isdir(model_dir)
        report = verify_qwen3_vl_checkpoint(model_dir)
        spec = qwen3_vl_checkpoint_spec()
        @test report.variant == :qwen3_vl_2b_instruct
        @test report.model_id == spec.model_id
        @test report.modelscope_revision == spec.modelscope_revision
        @test report.hf_revision == spec.hf_revision
        @test length(report.assets) == 13
        @test report.tensor_count == 625
        @test report.parameter_count == 2_127_532_032
        @test report.tensor_bytes == 4_255_064_064
        @test report.config.vision == spec.vision
        @test report.config.text == spec.text
        @test report.source == abspath(model_dir)
    end
end
