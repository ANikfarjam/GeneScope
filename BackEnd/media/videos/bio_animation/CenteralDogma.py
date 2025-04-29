from manim import *
import math
import random
from manim import Group

config.pixel_height = 1080  # 1080p resolution
config.pixel_width = 1920
config.frame_rate = 60  # Higher frame rates for smoother animations

class LungShape(VMobject):
    def __init__(self, color=WHITE, **kwargs):
        super().__init__(**kwargs)
        self.set_stroke(color=color, width=4)
        self.set_fill(opacity=0, color=color)  # Set opacity to 0 for no background
        
        # Left Lung (slightly smaller due to cardiac notch)
        self.start_new_path(np.array([-1.5, -1.0, 0]))
        self.add_cubic_bezier_curve_to(
            np.array([-2.2, 0.0, 0]),
            np.array([-2.2, 1.5, 0]),
            np.array([-1.8, 2.0, 0])
        )
        self.add_cubic_bezier_curve_to(
            np.array([-1.3, 2.3, 0]),
            np.array([-0.8, 2.0, 0]),
            np.array([-0.6, 1.0, 0]) 
        )
        # Cardiac notch (indent for heart)
        self.add_cubic_bezier_curve_to(
            np.array([-0.4, 0.3, 0]),
            np.array([-0.5, -0.5, 0]),
            np.array([-1.5, -1.0, 0])
        )
        
        # Right Lung (larger)
        self.start_new_path(np.array([0.6, -1.2, 0]))
        self.add_cubic_bezier_curve_to(
            np.array([1.5, -0.5, 0]),
            np.array([2.2, 0.5, 0]),
            np.array([2.2, 1.5, 0])
        )
        self.add_cubic_bezier_curve_to(
            np.array([2.0, 2.5, 0]),
            np.array([1.2, 2.6, 0]),
            np.array([0.8, 2.0, 0])
        )
        self.add_cubic_bezier_curve_to(
            np.array([0.5, 1.5, 0]),
            np.array([0.4, 0.0, 0]),
            np.array([0.6, -1.2, 0])
        )

class LiverShape(VMobject):
    def __init__(self, color=GRAY_BROWN, **kwargs):
        super().__init__(**kwargs)
        self.set_stroke(color=color, width=4)
        self.set_fill(opacity=0.1, color=color)
        
        self.start_new_path(np.array([-1.8, 0.0, 0]))
        self.add_cubic_bezier_curve_to(
            np.array([-1.0, 1.5, 0]),
            np.array([1.5, 1.5, 0]),
            np.array([2.0, 0.5, 0])
        )
        self.add_cubic_bezier_curve_to(
            np.array([1.0, 0.0, 0]),
            np.array([0.5, -0.5, 0]),
            np.array([-0.5, -0.3, 0])
        )
        self.add_cubic_bezier_curve_to(
            np.array([-1.2, -0.2, 0]),
            np.array([-1.8, -0.1, 0]),
            np.array([-1.8, 0.0, 0])
        )



class CancerClassification(ThreeDScene):
    def construct(self):
        #Scene 1: Intro
        # Scene 1: Introduction to Language Modeling
        title1 = Text("Breast Cancer Classification", font_size=32)
        subject = Text("BioMarkers and Prognosis").next_to(title1, DOWN)
        author= Text("Author: Ashkan Nikfarjam").next_to(subject, DOWN)
        self.play(Write(title1))
        self.play(Write(subject))
        self.play(FadeIn(author))
        self.wait(2)
        self.play(FadeOut(title1), FadeOut(subject), FadeOut(author))
        #Scene: Importance of genes
        intro = Text("Importance of Genes", font_size=28, color='GREEN_B').to_edge(UP)
        self.add(intro)
        self.wait(2)
        # DNA Properties
        num_pairs = 20  # Number of base pairs
        length = 0.5  # Distance between base pairs
        radius = 1.5  # Radius of the helix
        angle_step = PI / 10  # Angle step for each base pair

        # Colors for bases
        base_colors = {"A": BLUE, "T": ORANGE, "C": GREEN, "G": RED}
        base_pairs = ["A-T", "C-G", "G-C", "T-A"] * (num_pairs // 4)

        # Create the DNA strands
        left_strand = VGroup()
        right_strand = VGroup()
        base_connectors = VGroup()

        for i in range(num_pairs):
            angle = i * angle_step
            x = i * length - (num_pairs * length) / 2  # Horizontal placement
            y_left = radius * np.cos(angle) - 1  # Shift downward slightly
            z_left = radius * np.sin(angle)

            y_right = radius * np.cos(angle + PI) - 1
            z_right = radius * np.sin(angle + PI)

            # Create helical strands
            left_dot = Sphere(radius=0.1, color=WHITE).move_to([x, y_left, z_left])
            right_dot = Sphere(radius=0.1, color=WHITE).move_to([x, y_right, z_right])
            left_strand.add(left_dot)
            right_strand.add(right_dot)

            # Base connection lines
            base_pair = base_pairs[i]
            base1, base2 = base_pair.split("-")

            base_text_left = Text(base1, color=base_colors[base1]).scale(0.4).move_to([x, y_left, z_left])
            base_text_right = Text(base2, color=base_colors[base2]).scale(0.4).move_to([x, y_right, z_right])

            connector = Line(left_dot.get_center(), right_dot.get_center(), color=YELLOW)
            base_connectors.add(base_text_left, base_text_right, connector)

        # Group DNA elements together and position under the title
        dna_group = VGroup(left_strand, right_strand, base_connectors).move_to(DOWN * 1.5)

        # Animation
        self.play(Write(dna_group), run_time=3)

        # Rotation Animation (Optional, to make it look dynamic)
        self.play(Rotate(dna_group, angle=2 * PI, axis=RIGHT, about_point=ORIGIN, run_time=5))
        
        self.wait(2)
        self.play(FadeOut(dna_group))
        
        #Scene2
        # DNA Sequence Text
        dna_txt = Text("DNA", font_size=28, color=ORANGE).to_edge(LEFT).shift(UP * 2)
        sequence_str = "ATGCGTACGTTAGCTAGGCTTACGATCGATCGTAGCTAGCTAGGCTAG"
        sequence = Text(sequence_str, font_size=24).next_to(dna_txt, RIGHT)
        
        self.play(Write(dna_txt), Write(sequence))
        self.wait(2)

        # Select 4 random regions to highlight
        def get_non_overlapping_indices(sequence_length, region_length, count, min_spacing=2):
            indices = []
            attempts = 0
            while len(indices) < count and attempts < 1000:
                idx = random.randint(0, sequence_length - region_length)
                if all(abs(idx - existing) > region_length + min_spacing for existing in indices):
                    indices.append(idx)
                attempts += 1
            return sorted(indices)

        # Example usage:
        indices = get_non_overlapping_indices(len(sequence_str), 4, 4)
        colors = [RED, BLUE, GREEN, YELLOW]
        highlights = VGroup()

        for i, idx in enumerate(indices):
            # Extract substring and create a new Text object
            highlight_text = sequence_str[idx:idx+4]
            highlight = Text(highlight_text, font_size=24, color=colors[i])
            
            # Positioning the highlight correctly
            char_width = 0.17  # Approximate character width spacing
            highlight.next_to(dna_txt, DOWN).shift(RIGHT * idx * char_width)
            
            highlights.add(highlight)

        self.play(*[Transform(Text(sequence_str[idx:idx+3], font_size=24), highlights[i]) for i, idx in enumerate(indices)])
        self.wait(2)

        # Draw arrows and gene labels
        gene_boxes = VGroup()
        arrows = VGroup()

        previous_x = None
        vertical_shift = [DOWN * 2, DOWN * 2.5, DOWN * 2, DOWN * 2.5]  # alternate vertical positions

        for i, idx in enumerate(indices):
            gene_box = Text(f"Gene{i+1}", font_size=20, color=colors[i])
            
            # Position gene label
            gene_box.move_to(highlights[i].get_bottom() + vertical_shift[i])

            # Arrow from highlight to gene label
            arrow = Arrow(highlights[i].get_bottom(), gene_box.get_top(), color=colors[i], buff=0.1)

            gene_boxes.add(gene_box)
            arrows.add(arrow)

        self.play(*[GrowArrow(arrow) for arrow in arrows], *[Write(gene_box) for gene_box in gene_boxes])
        
        # Arrow to mRNA
        mRNA_text = Text("mRNA", font_size=28, color=WHITE).next_to(gene_boxes, RIGHT * 3)
        arrow_to_mRNA = Arrow(gene_boxes.get_right(), mRNA_text.get_left(), color=WHITE)

        self.play(GrowArrow(arrow_to_mRNA), Write(mRNA_text))
        self.wait(2)

        # Arrow to Protein
        protein_text = Text("Protein", font_size=28, color=GOLD).next_to(mRNA_text, RIGHT * 3)
        arrow_to_protein = Arrow(mRNA_text.get_right(), protein_text.get_left(), color=GOLD)

        self.play(GrowArrow(arrow_to_protein), Write(protein_text))
        self.wait(2)

        self.clear()

        human_img = ImageMobject('./HumanBody.png')
        human_img.scale(0.5)
        human_img.to_edge(LEFT)

        self.play(FadeIn(human_img))
        self.wait(2)

        #########option 1############
        # Create a cluster of small double helix DNA symbols
        # Create a cluster of small double helix DNA symbols (larger size version)
        gene_cluster = VGroup()
        num_genes = 10
        colors_list = [RED, GREEN, BLUE, ORANGE, PURPLE, TEAL, YELLOW]

        for i in range(num_genes):
            helix = VGroup()

            # Larger helix settings
            helix_height = 0.25  # increased from 0.1
            base_radius = 0.15   # increased from 0.1
            dot_radius = 0.05    # increased from 0.03
            num_base_pairs = 6   # a bit taller helix

            for j in range(num_base_pairs):
                angle = j * PI / 5
                y_offset = j * helix_height

                left = Dot3D(point=[-base_radius * np.cos(angle), y_offset, base_radius * np.sin(angle)],
                             radius=dot_radius, color=random.choice(colors_list))
                right = Dot3D(point=[base_radius * np.cos(angle), y_offset, -base_radius * np.sin(angle)],
                              radius=dot_radius, color=random.choice(colors_list))
                connector = Line3D(start=left.get_center(), end=right.get_center(), color=WHITE, stroke_width=1.5)

                helix.add(left, right, connector)

            # Random rotation and placement
            helix.rotate(random.uniform(0, TAU), axis=OUT)
            helix.rotate(random.uniform(0, TAU), axis=RIGHT)
            helix.shift(RIGHT * random.uniform(4.5, 6) + UP * random.uniform(-1.5, 2.5))

            gene_cluster.add(helix)

        # Add "23000+ genes" label
        # Move the gene label directly to the bottom of the gene cluster
        gene_cluster.next_to(human_img,RIGHT)
        gene_label = Text("23000+ genes", font_size=18, color=WHITE)  # Change to WHITE color
        gene_label.move_to(gene_cluster.get_bottom() + DOWN * 0.3)  # Position precisely under the genes

        # Show DNA gene cluster and label
        self.play(FadeIn(gene_cluster), Write(gene_label))
        self.wait(3)

        #########option 2############
                # Gene Cluster: Double Helix representation
        # def create_double_helix(radius=0.2, height=1.5, turns=2, color=BLUE):
        #     num_segments = 20
        #     base_group = VGroup()
        #     for i in range(num_segments):
        #         angle = 2 * PI * turns * i / num_segments
        #         y = height * i / num_segments
        #         x1 = radius * math.cos(angle)
        #         z1 = radius * math.sin(angle)
        #         x2 = radius * math.cos(angle + PI)
        #         z2 = radius * math.sin(angle + PI)

        #         base1 = Sphere(radius=0.05, color=color).move_to([x1, y, z1])
        #         base2 = Sphere(radius=0.05, color=color).move_to([x2, y, z2])
        #         connector = Line(base1.get_center(), base2.get_center(), color=WHITE, stroke_width=1)
        #         base_group.add(base1, base2, connector)
        #     return base_group

        # cluster = VGroup()
        # cluster_colors = [RED, BLUE, GREEN, YELLOW, ORANGE, PURPLE, PINK, TEAL, MAROON, GOLD]

        # for i in range(20):  # create 20 mini helices
        #     helix = create_double_helix(color=random.choice(cluster_colors))
        #     helix.scale(0.5)
        #     # Random rotation
        #     helix.rotate(angle=random.uniform(0, 2 * PI), axis=random.choice([RIGHT, UP, OUT]))
        #     # Position them randomly but grouped
        #     helix.shift(RIGHT * random.uniform(3.5, 5.5) + UP * random.uniform(-1, 2))
        #     cluster.add(helix)
        # cluster.next_to(human_img, RIGHT)
        # # Add the cluster to the scene
        # self.play(FadeIn(cluster), run_time=3)

        # # Label for the cluster
        # gene_label = Text("23000+ genes", font_size=20, color=WHITE)
        # gene_label.next_to(cluster, DOWN, buff=0.5)
        # self.play(Write(gene_label))
        # self.wait(3)


        #########adign origanisims shape######
        def create_small_helix():
            helix = VGroup()
            helix_height = 0.25
            base_radius = 0.15
            dot_radius = 0.05
            num_base_pairs = 6
            colors_list = [RED, GREEN, BLUE, ORANGE, PURPLE, TEAL, YELLOW]
            for j in range(num_base_pairs):
                angle = j * PI / 5
                y_offset = j * helix_height

                left = Dot3D(point=[-base_radius * np.cos(angle), y_offset, base_radius * np.sin(angle)],
                            radius=dot_radius, color=random.choice(colors_list))
                right = Dot3D(point=[base_radius * np.cos(angle), y_offset, -base_radius * np.sin(angle)],
                            radius=dot_radius, color=random.choice(colors_list))
                connector = Line3D(start=left.get_center(), end=right.get_center(), color=WHITE, stroke_width=1.5)

                helix.add(left, right, connector)
            return helix

        # First create Lung group
        lung_helix = create_small_helix().scale(0.7)

        lung_genes = VGroup(
            Text('TP53', font_size=20),
            Text('EGFR', font_size=20),
            Text('KRAS', font_size=20)
        ).arrange(DOWN, buff=0.1)

        lung_left = VGroup(lung_helix, lung_genes).arrange(DOWN, buff=0.3)

        lung_img = ImageMobject('./lung.png').scale(0.4)
        lung_name = Text('Lung', font_size=24)

        lung_right = Group(lung_img, lung_name).arrange(DOWN, buff=0.3)

        lung_group = Group(lung_left, lung_right).arrange(RIGHT, buff=1.0)

        # Now create Liver group
        liver_helix = create_small_helix().scale(0.7)

        liver_genes = VGroup(
            Text('ALB', font_size=20),
            Text('AFP', font_size=20),
            Text('CYP3A4', font_size=20)
        ).arrange(DOWN, buff=0.1)

        liver_left = VGroup(liver_helix, liver_genes).arrange(DOWN, buff=0.3)

        liver_img = ImageMobject('./liver.png').scale(0.4)
        liver_name = Text('Liver', font_size=24)

        liver_right = Group(liver_img, liver_name).arrange(DOWN, buff=0.3)

        liver_group = Group(liver_left, liver_right).arrange(RIGHT, buff=1.0)

        # Group both together horizontally
        both_groups = Group(lung_group, liver_group).arrange(DOWN, buff=0.3)

        both_groups.scale(0.9)  # Just to be safe, fit screen
        both_groups.to_edge(RIGHT)  # Lower it a bit for nice positioning

        # Animate
        self.play(FadeIn(both_groups))
        self.wait(3)
        self.clear()

        ####mutation scene###
        # Gene blocks (normal DNA)
        # Step 1: Normal genes
        gene_names = ["TP53", "BRCA1", "EGFR", "KRAS", "MYC"]
        genes = VGroup(*[
            Rectangle(width=1.2, height=0.5, color=BLUE).set_fill(BLUE, opacity=0.5)
            for _ in gene_names
        ])
        for gene, name in zip(genes, gene_names):
            label = Text(name, font_size=24).move_to(gene)
            gene.add(label)

        genes.arrange(RIGHT, buff=0.3)
        normal_label = Text("Normal DNA", font_size=30).next_to(genes, UP)

        self.play(FadeIn(genes), Write(normal_label))
        self.wait(1)

        # Step 2: Mutated genes
        mutated_genes = genes.copy()
        self.play(FadeOut(genes))  # 🛠️ Fade out old original genes immediately!

        # 2.1 Missing Gene: remove BRCA1 (index 1)
        missing_gene = mutated_genes[1]
        self.play(FadeOut(missing_gene))
        mutated_genes.remove(missing_gene)
        self.play(mutated_genes.animate.arrange(RIGHT, buff=0.3))
        self.wait(1)

        # 2.2 Wrong Gene: TP53 -> BAD1
        wrong_gene = mutated_genes[0]
        new_label = Text("BAD1", font_size=24).move_to(wrong_gene)
        self.play(
            Transform(wrong_gene[1], new_label),
            wrong_gene.animate.set_color(RED)
        )
        self.wait(1)

        # 2.3 Overexpression: duplicate MYC (last gene)
        overexpressed = mutated_genes[-1].copy()
        overexpressed.shift(RIGHT * 1.5)
        overexpressed.set_color(ORANGE)

        mutated_group = VGroup(*mutated_genes, overexpressed)

        self.play(FadeIn(overexpressed))
        self.play(mutated_group.animate.arrange(RIGHT, buff=0.3))
        self.wait(2)

        # Step 3: Add Mutated DNA Label
        mutated_label = Text("Mutated DNA", font_size=30).next_to(mutated_group, UP)
        self.play(Write(mutated_label))
        self.wait(2)