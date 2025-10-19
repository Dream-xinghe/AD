import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from shapely import affinity, ops
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString

CLASS2LABEL = {
    'road_divider': 0,
    'lane_divider': 0,
    'ped_crossing': 1,
    'contours': 2,
    'others': -1
}

class VectorizedLocalMap(object):
    def __init__(self,
                 dataroot,
                 patch_size,
                 canvas_size,
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment', 'lane'],
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 normalize=False,
                 fixed_num=-1):
        '''
        Args:
            fixed_num = -1 : no fixed num
        '''
        super().__init__()
        self.data_root = dataroot
        self.MAPS = ['boston-seaport', 'singapore-hollandvillage',
                     'singapore-onenorth', 'singapore-queenstown']
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            # NuScenesMap用于加载全局地图数据
            self.nusc_maps[loc] = NuScenesMap(dataroot=self.data_root, map_name=loc)
            # NuScenesMapExplorer用于在全局地图中提取局部地图的几何信息
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])

        self.patch_size = patch_size
        self.canvas_size = canvas_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.normalize = normalize
        self.fixed_num = fixed_num

    def gen_vectorized_samples(self, location, ego2global_translation, ego2global_rotation):
        # map_pose代表的是当前车在全局地图中的位置(x, y)
        map_pose = ego2global_translation[:2]
        # 经过Quaternion转换后，rotation代表的是当前车在全局地图中的朝向
        rotation = Quaternion(ego2global_rotation)

        # 生成局部地图的patch_box和patch_angle，patch_size=(102.4, 102.4)
        # patch_box代表以自车为中心的局部地图范围(x, y, w, l)
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        # patch_angle代表自车在全局坐标系中的朝向角
        patch_angle = quaternion_yaw(rotation) / np.pi * 180

        # line_geom储存的形式为[(layer_name, geoms), ...]
        # 获取数据集中该location下的地图元素的几何信息
        line_geom = self.get_map_geom(patch_box, patch_angle, self.line_classes, location)
        # line_vector_list代表线类型地图元素的矢量化表示
        line_vector_dict = self.line_geoms_to_vectors(line_geom)

        # 获取人行横道的几何信息,在get_ped_crossing_line将人行道的几何信息转换成线段表示
        ped_geom = self.get_map_geom(patch_box, patch_angle, self.ped_crossing_classes, location)
        # ped_vector_list = self.ped_geoms_to_vectors(ped_geom)
        # 将人行道和line类型的地图元素使用同样的函数进行矢量化表示
        ped_vector_list = self.line_geoms_to_vectors(ped_geom)['ped_crossing']

        # 获取地图里的road-segment和lane的几何信息
        polygon_geom = self.get_map_geom(patch_box, patch_angle, self.polygon_classes, location)
        # 使用poly_geoms_to_vectors将面类型地图元素的几何信息转换内外轮廓的线段表示之后再转换成成矢量表示
        poly_bound_list = self.poly_geoms_to_vectors(polygon_geom)

        vectors = []
        for line_type, vects in line_vector_dict.items():
            for line, length in vects:
                vectors.append((line.astype(float), length, CLASS2LABEL.get(line_type, -1)))

        for ped_line, length in ped_vector_list:
            vectors.append((ped_line.astype(float), length, CLASS2LABEL.get('ped_crossing', -1)))

        for contour, length in poly_bound_list:
            vectors.append((contour.astype(float), length, CLASS2LABEL.get('contours', -1)))

        # filter out -1
        filtered_vectors = []
        for pts, pts_num, type in vectors:
            if type != -1:
                filtered_vectors.append({
                    'pts': pts,
                    'pts_num': pts_num,
                    'type': type
                })

        return filtered_vectors

    def get_map_geom(self, patch_box, patch_angle, layer_names, location):
        map_geom = []
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                # 如果是线类型的地图元素，调用_get_layer_line方法获取几何信息，geoms是一个LineString列表
                geoms = self.map_explorer[location]._get_layer_line(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.polygon_classes:
                # 如果是面类型的地图元素，调用_get_layer_polygon方法获取几何信息，geoms是一个Polygon列表
                geoms = self.map_explorer[location]._get_layer_polygon(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.ped_crossing_classes:
                # 对于人行道的元素，最后返回的是两条最长边代表的线段列表
                geoms = self.get_ped_crossing_line(patch_box, patch_angle, location)
                # geoms = self.map_explorer[location]._get_layer_polygon(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
        return map_geom

    def _one_type_line_geom_to_vectors(self, line_geom):
        line_vectors = []
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_vectors.append(self.sample_pts_from_line(single_line))
                elif line.geom_type == 'LineString':
                    line_vectors.append(self.sample_pts_from_line(line))
                else:
                    raise NotImplementedError
        # line_vectors储存的形式为[(sampled_points, num_valid), ...]
        # line_vectors代表线类型地图元素的矢量化表示，即采样点坐标和有效采样点数量
        return line_vectors

    def poly_geoms_to_vectors(self, polygon_geom):
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        # 将所有的road-segment和lane的多边形进行合并，去掉重合的区域
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        # union_segments储存的是合并后的road-segment和lane的多边形
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        # local_patch代表局部地图的范围
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        # 提取多边形的外轮廓和内轮廓
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        # 这里通过is_ccw判断线段的连通方向，也可以用于区分外轮廓和内轮廓
        for ext in exteriors:
            # 保证外轮廓是逆时针方向,ext.is_ccw表示是否为逆时针方向
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            # 进行裁剪，提取在局部地图范围内的线段
            lines = ext.intersection(local_patch)
            # 如果是多条不连通的线段，进行合并
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def line_geoms_to_vectors(self, line_geom):
        line_vectors_dict = dict()
        for line_type, a_type_of_lines in line_geom:
            # one_type_vectors储存的形式为[(sampled_points, num_valid), ...]
            one_type_vectors = self._one_type_line_geom_to_vectors(a_type_of_lines)
            line_vectors_dict[line_type] = one_type_vectors

        return line_vectors_dict

    def ped_geoms_to_vectors(self, ped_geom):
        ped_geom = ped_geom[0][1]
        union_ped = ops.unary_union(ped_geom)
        if union_ped.geom_type != 'MultiPolygon':
            union_ped = MultiPolygon([union_ped])

        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        results = []
        for ped_poly in union_ped:
            # rect = ped_poly.minimum_rotated_rectangle
            ext = ped_poly.exterior
            if not ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def get_ped_crossing_line(self, patch_box, patch_angle, location):
        # 将两条最长边提取出来并且裁剪到patch范围内
        def add_line(poly_xy, idx, patch, patch_angle, patch_x, patch_y, line_list):
            # 取出来要选取线段的两个端点
            points = [(p0, p1) for p0, p1 in zip(poly_xy[0, idx:idx + 2], poly_xy[1, idx:idx + 2])]
            line = LineString(points)
            # intersection用于裁剪线段到patch范围内
            line = line.intersection(patch)
            # 先裁剪再进行旋转和平移
            if not line.is_empty:
                line = affinity.rotate(line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                line = affinity.affine_transform(line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(line)

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        # patch表示局部地图的范围，使用get_patch_coord函数获取的坐标是一个shapely的Polygon对象
        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        line_list = []
        # records表示该location下所有的ped_crossing元素的信息列表，即返回ped_crossing这一图层的所有元素
        records = getattr(self.nusc_maps[location], 'ped_crossing')
        for record in records:
            # 通过polygon_token提取ped_crossing元素的几何信息，polygon是一个shapely的Polygon对象
            polygon = self.map_explorer[location].extract_polygon(record['polygon_token'])
            # 一个环形的(x,y)坐标数组
            poly_xy = np.array(polygon.exterior.xy)
            # 计算没两点代表的线段长度
            dist = np.square(poly_xy[:, 1:] - poly_xy[:, :-1]).sum(0)
            x1, x2 = np.argsort(dist)[-2:]

            # 添加两条线段到line_list中
            add_line(poly_xy, x1, patch, patch_angle, patch_x, patch_y, line_list)
            add_line(poly_xy, x2, patch, patch_angle, patch_x, patch_y, line_list)

        return line_list

    def sample_pts_from_line(self, line):
        # 如果未指定固定采样点数量，则按照固定距离采样
        if self.fixed_num < 0:
            distances = np.arange(0, line.length, self.sample_dist)
            # 根据距离在line上进行插值，得到采样点的坐标
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
        else:
            # fixed number of points, so distance is line.length / self.fixed_num
            distances = np.linspace(0, line.length, self.fixed_num)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)

        # 如果需要归一化将点坐标到[0, 1]范围内
        if self.normalize:
            sampled_points = sampled_points / np.array([self.patch_size[1], self.patch_size[0]])

        num_valid = len(sampled_points)

        if not self.padding or self.fixed_num > 0:
            # fixed num sample can return now!
            return sampled_points, num_valid

        # fixed distance sampling need padding!
        num_valid = len(sampled_points)

        if self.fixed_num < 0:
            if num_valid < self.num_samples:
                padding = np.zeros((self.num_samples - len(sampled_points), 2))
                sampled_points = np.concatenate([sampled_points, padding], axis=0)
            else:
                sampled_points = sampled_points[:self.num_samples, :]
                num_valid = self.num_samples

            if self.normalize:
                sampled_points = sampled_points / np.array([self.patch_size[1], self.patch_size[0]])
                num_valid = len(sampled_points)

        # 如果不需要padding或者已经是固定数量的采样点，则直接返回
        # 否则进行padding或者截断操作
        return sampled_points, num_valid
