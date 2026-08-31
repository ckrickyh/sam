from segment_canopy_leaves import segment_canopy_leaves


def main():
    results = segment_canopy_leaves(
        image_path="data/test01.jpeg",
        checkpoint_path="sam3.pt",
        crown_prompt="foliage cluster",
        prompt_text="green tree leaves",
        trunk_prompt="tree trunk, branches",
        output_path="data/test01_leaves_segmented.png",
        threshold=0.30,
        enable_color_filter=True,
        isolate_main_tree=True,
    )
    print(f"計算完成！")
    print(f"- 樹冠整體面積 (Crown Area 分母): {results['crown_area_pixels']:,} px")
    print(f"- 純葉片覆蓋面積 (分子): {results['pure_leaf_pixels']:,} px")
    print(f"- 樹冠內葉片密度 (Crown Leaf Density): {results['crown_leaf_density_percentage']:.2f}%")
    print(f"- 樹冠透光/孔隙率 (Crown Porosity): {results['crown_porosity_percentage']:.2f}%")


if __name__ == "__main__":
    main()
