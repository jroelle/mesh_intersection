#define IGL_ALG 0
#define TRY_CATCH 1

#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>

#if IGL_ALG
#include <igl/copyleft/cgal/mesh_boolean.h>
#else
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/intersections.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#endif

#include <Eigen/Core>

#include <iostream>
#include <chrono>

#include "triangle_intersection.h"
#include <string>

igl::opengl::glfw::Viewer viewer;

Eigen::MatrixXd VA, VB, VC;
#if IGL_ALG
Eigen::VectorXi J, I;
#endif
Eigen::MatrixXi FA, FB, FC;

constexpr double EPS = 1e-6;

using K = CGAL::Simple_cartesian<float>; /*CGAL::Exact_predicates_inexact_constructions_kernel;*/
using T = /*CGAL::Constrained_Delaunay_triangulation_2<K>;*/ CGAL::Constrained_triangulation_2<K, CGAL::Default, CGAL::Exact_predicates_tag>; /*CGAL::Triangulation_3<K>;*/

namespace CGAL
{
	template<>
	struct Compare<K::RT>
	{
		using result_type = Sign;

		Comparison_result operator()(K::RT x, K::RT y) const
		{
			if (abs(x - y) < EPS)
				return EQUAL;
			return x < y ? SMALLER : LARGER;
		}
	};

	template<>
	typename Compare<K::RT>::result_type compare(const K::RT &a, const K::RT &b)
	{
		Compare<K::RT> comparator;
		return comparator(a, b);
	}
}


int a_id = -1;
int b_id = -1;
int c_id = -1;

enum class SimpleObjects : uint8_t
{
	None,
	Cube,
	Triangles
};

constexpr auto SIMPLE_OBJECTS = SimpleObjects::None;


struct Vector
{
	double x = 0;
	double y = 0;
	double z = 0;

	constexpr Vector(const double v[3]) :
		x(v[0]),
		y(v[1]),
		z(v[2])
	{}
	constexpr Vector(const float v[3]) :
		x(v[0]),
		y(v[1]),
		z(v[2])
	{}
	constexpr Vector(double X, double Y, double Z) :
		x(X),
		y(Y),
		z(Z)
	{}
	template<typename V>
	inline Vector(const V &v) :
		x(v.x()),
		y(v.y()),
		z(v.z())
	{}
	constexpr Vector(const Vector &) = default;
	constexpr Vector() = default;

	constexpr double Dot(const Vector &other) const
	{
		return x * other.x + y * other.y + z * other.z;
	}
	constexpr Vector Cross(const Vector &other) const
	{
		return { y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x };
	}
	Vector operator*(double k) const
	{
		return { x * k, y * k, z * k };
	}
	Vector operator+(const Vector &other) const
	{
		return { x + other.x, y + other.y, z + other.z };
	}
	Vector operator-(const Vector &other) const
	{
		return { x - other.x, y - other.y, z - other.z };
	}
	Vector Normalize() const
	{
		const double length = sqrt(this->Dot(*this));
		return { x / length, y / length, z / length };
	}
	friend std::ostream &operator<<(std::ostream &os, const Vector &v)
	{
		os << "X: " << v.x << ", Y: " << v.y << ", Z: " << v.z;
		return os;
	}
};

using Point = Vector;
struct Triangle
{
	Point v1;
	Point v2;
	Point v3;

	constexpr Triangle() = default;
	constexpr Triangle(const Point &p1, const Point &p2, const Point &p3) :
		v1(p1),
		v2(p2),
		v3(p3)
	{}
	friend std::ostream &operator<<(std::ostream &os, const Triangle &t)
	{
		os << "V1: " << t.v1 << std::endl << "V2: " << t.v2 << std::endl << "V3: " << t.v3 << std::endl;
		return os;
	}
};


constexpr auto INTERSECTION_COLOR = Vector{ 0.5, 0.5, 1 };
constexpr auto INTERSECTION_ERROR_COLOR = Vector{ 0, 1, 0 };
constexpr auto INTERSECTION_NOT_FOUND_COLOR = Vector{ 1, 0, 1 };
constexpr auto TRIANGULATION_ERROR_COLOR = Vector{ 1, 0, 0 };
constexpr auto TRIANGULATION_HANDLED_COLOR = Vector{ 0, 1, 0 };


static inline bool valid(const K::Triangle_3 &t1, const K::Triangle_3 &t2)
{
	return CGAL::intersection(t1.supporting_plane(), t2).has_value() ||
		CGAL::intersection(t2.supporting_plane(), t1).has_value();
}

static inline double length(const K::Vector_3 &v)
{
	return sqrt(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
}

static inline bool is_valid(const K::Triangle_3 &t)
{
	const auto is_degenerate = t.squared_area() < std::numeric_limits<double>::epsilon();
	return !is_degenerate;
}

static inline bool triangle_intersects_plane(const Triangle &triangle, const Triangle &plane)
{
	const auto vb1 = plane.v1;
	const auto vb2 = plane.v2;
	const auto vb3 = plane.v3;

	const auto n = (vb2 - vb1).Cross(vb3 - vb1).Normalize();
	const auto d = (n * -1).Dot(vb1);

	Vector dist;

	dist.x = n.Dot(triangle.v1) + d;
	dist.y = n.Dot(triangle.v2) + d;
	dist.z = n.Dot(triangle.v3) + d;

	bool points_on_same_side = true;

	if (abs(dist.x) >= EPS || abs(dist.y) >= EPS || abs(dist.z) >= EPS)
	{
		points_on_same_side = dist.x >= 0 && dist.y >= 0 && dist.z >= 0 ||
			dist.x < 0 && dist.y < 0 && dist.z < 0;
	}
	else
	{
		//coplanar
	}

	return !points_on_same_side;
}

void merge(Eigen::MatrixXd &VA, Eigen::MatrixXi &FA, const Eigen::MatrixXd &VB, const Eigen::MatrixXi &FB)
{
	if (FA.rows() > 0)
	{
		Eigen::Matrix<size_t, 2, 1> sizes(FA.rows(), FB.rows());
		Eigen::Matrix<Eigen::MatrixXd::Scalar, Eigen::Dynamic, 3> VV(VA.rows() + VB.rows(), 3);
		Eigen::MatrixXi FF(FA.rows() + FB.rows(), 3);
		for (int a = 0; a < VA.rows(); a++)
		{
			for (int d = 0; d < 3; d++)
				VV(a, d) = VA(a, d);
		}
		for (int b = 0; b < VB.rows(); b++)
		{
			for (int d = 0; d < 3; d++)
				VV(VA.rows() + b, d) = VB(b, d);
		}
		FF.block(0, 0, FA.rows(), 3) = FA;
		if (FB.rows() > 0)
		{
			FF.block(FA.rows(), 0, FB.rows(), 3) = FB.array() + VA.rows();
		}
		VA = VV;
		FA = FF;
	}
	else
	{
		VA = VB;
		FA = FB;
	}
}

void draw_point(const K::Point_3 &p, int data_id, const Vector &color)
{
	const Eigen::MatrixXd v = (Eigen::MatrixXd(1, 3) <<
		p.x(), p.y(), p.z()).finished();

	const Eigen::MatrixXd c = (Eigen::MatrixXd(1, 3) <<
		color.x, color.y, color.z).finished();

	viewer.data(data_id).add_points(v, c);
}

void draw_edge(const K::Point_3 &p1, const K::Point_3 &p2, int data_id, const Vector &color)
{
	const Eigen::MatrixXd c = (Eigen::MatrixXd(1, 3) <<
		color.x, color.y, color.z).finished();

	const Eigen::MatrixXd v1 = (Eigen::MatrixXd(1, 3) <<
		p1.x(), p1.y(), p1.z()).finished();

	const Eigen::MatrixXd v2 = (Eigen::MatrixXd(1, 3) <<
		p2.x(), p2.y(), p2.z()).finished();

	viewer.data(data_id).add_edges(v1, v2, c);
}

void draw_edges(const K::Triangle_3 &t, int data_id, const Vector &color)
{
	draw_edge(t.vertex(0), t.vertex(1), data_id, color);
	draw_edge(t.vertex(0), t.vertex(2), data_id, color);
	draw_edge(t.vertex(1), t.vertex(2), data_id, color);
};

template<typename T>
void write_time(std::string_view str, const T &time)
{
	std::cout << str << std::chrono::duration_cast<std::chrono::nanoseconds>(time).count() * 1.e-6 << " ms" << std::endl;
}

class IntersectionVisitor
{
public:
	using Intersections = std::map<std::pair<int, bool>, std::vector<K::Segment_3>>;

	explicit IntersectionVisitor(Intersections &intersections, const boost::optional<K::Aff_transformation_3> &scale, int row_a, int row_b) :
		_intersections(intersections),
		_scale(scale),
		_rows(std::make_pair(row_a, row_b))
	{}

	void operator()(const K::Point_3 &point)
	{
		const auto &p = without_scale(point);
	}

	void operator()(const K::Segment_3 &segment)
	{
		const auto &s = without_scale(segment);

		_intersections[std::make_pair(_rows.first, true)].emplace_back(s);
		_intersections[std::make_pair(_rows.second, false)].emplace_back(s);

		draw_edge(s.start(), s.end(), viewer.append_mesh(), INTERSECTION_COLOR);
	}

	void operator()(const K::Triangle_3 &triangle)
	{
		const auto &t = without_scale(triangle);
	}

	void operator()(const std::vector<K::Point_3> &points)
	{
		std::vector<K::Point_3> ps;
		ps.reserve(points.size());
	}

private:
	template<typename Object>
	Object without_scale(const Object &obj) const
	{
		if (_scale.has_value())
			return obj.transform(_scale->inverse());
		else
			return obj;
	}

private:
	Intersections &_intersections;
	const boost::optional<K::Aff_transformation_3> &_scale;
	std::pair<int, int> _rows;
};

std::map<std::pair<int, bool>, std::vector<K::Segment_3>> calculate_intersections(const Eigen::MatrixXd &VA, const Eigen::MatrixXi &FA, const Eigen::MatrixXd &VB, const Eigen::MatrixXi &FB)
{
	auto intersection_time = std::chrono::nanoseconds::zero();

	std::map<std::pair<int, bool>, std::vector<K::Segment_3>> intersections;
	for (Eigen::Index row_a = 0; row_a < FA.rows(); ++row_a)
	{
		const auto &verticesA = FA.row(row_a);

		const auto &va1i = FA(row_a, 0);
		const auto &va2i = FA(row_a, 1);
		const auto &va3i = FA(row_a, 2);

		const auto va1 = Vector{ VA(va1i, 0), VA(va1i, 1), VA(va1i, 2) };
		const auto va2 = Vector{ VA(va2i, 0), VA(va2i, 1), VA(va2i, 2) };
		const auto va3 = Vector{ VA(va3i, 0), VA(va3i, 1), VA(va3i, 2) };

		const Triangle ta{ va1, va2, va3 };

		for (Eigen::Index row_b = 0; row_b < FB.rows(); ++row_b)
		{
			const auto &vb1i = FB(row_b, 0);
			const auto &vb2i = FB(row_b, 1);
			const auto &vb3i = FB(row_b, 2);

			const auto vb1 = Vector{ VB(vb1i, 0), VB(vb1i, 1), VB(vb1i, 2) };
			const auto vb2 = Vector{ VB(vb2i, 0), VB(vb2i, 1), VB(vb2i, 2) };
			const auto vb3 = Vector{ VB(vb3i, 0), VB(vb3i, 1), VB(vb3i, 2) };

			const Triangle tb{ vb1, vb2, vb3 };

#if 1
			auto begin = std::chrono::steady_clock::now();

			K::Triangle_3 cgal_ta
			{
				K::Point_3(va1.x, va1.y, va1.z),
				K::Point_3(va2.x, va2.y, va2.z),
				K::Point_3(va3.x, va3.y, va3.z)
			};

			K::Triangle_3 cgal_tb
			{
				K::Point_3(vb1.x, vb1.y, vb1.z),
				K::Point_3(vb2.x, vb2.y, vb2.z),
				K::Point_3(vb3.x, vb3.y, vb3.z)
			};

#if TRY_CATCH && _DEBUG
			try
			{
#endif
				if (!triangle_intersects_plane(ta, tb) || !triangle_intersects_plane(tb, ta))
					continue;

				//if (!valid(cgal_ta, cgal_tb))
				//	continue;

				K::Triangle_3 t_a, t_b;

				const auto plane_a = cgal_ta.supporting_plane();
				const auto plane_b = cgal_ta.supporting_plane();

				constexpr double SMALL = 1.e-3;
				boost::optional<K::Aff_transformation_3> scale;

				if (abs(plane_a.a()) < SMALL && abs(plane_a.b()) < SMALL && abs(plane_a.c()) < SMALL && abs(plane_a.d()) < SMALL &&
					abs(plane_b.a()) < SMALL && abs(plane_b.b()) < SMALL && abs(plane_b.c()) < SMALL && abs(plane_b.d()) < SMALL)
				{
					scale.emplace(CGAL::Scaling{}, 1, SMALL);
					t_a = cgal_ta.transform(*scale);
					t_b = cgal_tb.transform(*scale);
				}
				else
				{
					t_a = cgal_ta;
					t_b = cgal_tb;
				}

				auto result = CGAL::intersection(t_a, t_b);

#if 0
				if (!result)
				{
					float v0[3] = { cgal_ta.vertex(0).x(), cgal_ta.vertex(0).y(), cgal_ta.vertex(0).z() };
					float v1[3] = { cgal_ta.vertex(1).x(), cgal_ta.vertex(1).y(), cgal_ta.vertex(1).z() };
					float v2[3] = { cgal_ta.vertex(2).x(), cgal_ta.vertex(2).y(), cgal_ta.vertex(2).z() };

					float u0[3] = { cgal_tb.vertex(0).x(), cgal_tb.vertex(0).y(), cgal_tb.vertex(0).z() };
					float u1[3] = { cgal_tb.vertex(1).x(), cgal_tb.vertex(1).y(), cgal_tb.vertex(1).z() };
					float u2[3] = { cgal_tb.vertex(2).x(), cgal_tb.vertex(2).y(), cgal_tb.vertex(2).z() };

					int coplanar = FALSE;
					float start[3] = { 0 };
					float end[3] = { 0 };
					if (tri_tri_intersect_with_isectline(v0, v1, v2, u0, u1, u2, &coplanar, start, end))
					{
						result.emplace(K::Segment_3(K::Point_3{ start[0], start[1], start[2] }, K::Point_3{ end[0], end[1], end[2] }));
					}
				}
#endif

				auto end = std::chrono::steady_clock::now();
				intersection_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

				if (result)
					boost::apply_visitor(IntersectionVisitor(intersections, scale, row_a, row_b), *result);
#if 0
				constexpr bool OUTPUT = false;
				if (result)
				{
					if (const auto *point = boost::get<K::Point_3>(&*result))
					{
						if (OUTPUT)
							std::cout << "POINT " << *point << std::endl;

						//draw_point(*point, viewer.append_mesh(), INTERSECTION_COLOR);
					}
					else if (const auto *segment = boost::get<K::Segment_3>(&*result))
					{
						if (OUTPUT)
							std::cout << "SEGMENT " << *segment << std::endl;

						draw_edge(segment->start(), segment->end(), viewer.append_mesh(), INTERSECTION_COLOR);

						intersections[std::make_pair(row_a, true)].emplace_back(*segment);
						intersections[std::make_pair(row_b, false)].emplace_back(*segment);
					}
					else if (const auto *triangle = boost::get<K::Triangle_3>(&*result))
					{
						if (OUTPUT)
							std::cout << "TRIANGLE " << *triangle << std::endl;

						draw_edges(*triangle, viewer.append_mesh(), INTERSECTION_COLOR);
					}
					else if (const auto *points = boost::get<std::vector<K::Point_3>>(&*result))
					{
						if (OUTPUT)
						{
							std::cout << "POINTS" << std::endl;
							for (const auto &point : *points)
								std::cout << "POINT " << point << std::endl;
						}

						//int id = viewer.append_mesh();
						//for (const auto &point : *points)
						//	draw_point(point, id, INTERSECTION_COLOR);
					}
				}
#endif
#if 0
				if (!result)
				{
					float v0[3] = { cgal_ta.vertex(0).x(), cgal_ta.vertex(0).y(), cgal_ta.vertex(0).z() };
					float v1[3] = { cgal_ta.vertex(1).x(), cgal_ta.vertex(1).y(), cgal_ta.vertex(1).z() };
					float v2[3] = { cgal_ta.vertex(2).x(), cgal_ta.vertex(2).y(), cgal_ta.vertex(2).z() };

					float u0[3] = { cgal_tb.vertex(0).x(), cgal_tb.vertex(0).y(), cgal_tb.vertex(0).z() };
					float u1[3] = { cgal_tb.vertex(1).x(), cgal_tb.vertex(1).y(), cgal_tb.vertex(1).z() };
					float u2[3] = { cgal_tb.vertex(2).x(), cgal_tb.vertex(2).y(), cgal_tb.vertex(2).z() };

					if (NoDivTriTriIsect(v0, v1, v2, u0, u1, u2))
					{
						const Vector color{ (double)rand() / RAND_MAX, (double)rand() / RAND_MAX, (double)rand() / RAND_MAX };

						std::cout << "NO INTERSECTION:" << std::endl <<
							"COLOR: " << color << std::endl << 
							row_a << std::endl <<
							cgal_ta.vertex(0) << std::endl <<
							cgal_ta.vertex(1) << std::endl <<
							cgal_ta.vertex(2) << std::endl << std::endl <<
							row_b << std::endl <<
							cgal_tb.vertex(0) << std::endl <<
							cgal_tb.vertex(1) << std::endl <<
							cgal_tb.vertex(2) << std::endl << std::endl << std::endl;

						int id = viewer.append_mesh();
						draw_edges(cgal_ta, id, color);
						draw_edges(cgal_tb, id, color);
					}
				}
#endif

#if TRY_CATCH && _DEBUG
			}
			catch (...)
			{
				std::cout << "INTERSECTION ERROR:" << std::endl <<
					row_a << std::endl <<
					cgal_ta.vertex(0) << std::endl <<
					cgal_ta.vertex(1) << std::endl <<
					cgal_ta.vertex(2) << std::endl << std::endl <<
					row_b << std::endl <<
					cgal_tb.vertex(0) << std::endl <<
					cgal_tb.vertex(1) << std::endl <<
					cgal_tb.vertex(2) << std::endl << std::endl << std::endl;

				int id = viewer.append_mesh();
				draw_edges(cgal_ta, id, INTERSECTION_ERROR_COLOR);
				draw_edges(cgal_tb, id, INTERSECTION_ERROR_COLOR);
			}
#endif

#endif
		}
	}

	write_time("intersection time: ", intersection_time);
	return intersections;
}

static inline K::Vector_3 get_normal(const K::Triangle_3 &t)
{
	const auto &v0 = t.vertex(0);
	const auto &v1 = t.vertex(1);
	const auto &v2 = t.vertex(2);

	const auto &n = Vector{ v1 - v0 }.Cross({ v2 - v0 }).Normalize();
	return { n.x, n.y, n.z };
}

static inline void get_triangle_lcs(const K::Triangle_3 &t, K::Vector_3 &e1, K::Vector_3 &e2, K::Vector_3 &e3, K::Point_3 &o)
{
	const Vector v0{ t.vertex(0) };
	const Vector v1{ t.vertex(1) };
	const Vector v2{ t.vertex(2) };

	const Vector x = (v1 - v0).Normalize();
	const auto n = x.Cross(v2 - v0).Normalize();
	const double d = (n * -1).Dot(v0);
	const Vector y = n.Cross(x);

	e1 = K::Vector_3{ x.x, x.y, x.z };
	e2 = K::Vector_3{ y.x, y.y, y.z };
	e3 = K::Vector_3{ n.x, n.y, n.z };
	o = K::Point_3{ v0.x, v0.y, v0.z };
}

static inline K::Point_3 get_point_in_triangle_lcs(const K::Point_3 &point, const K::Vector_3 &e1, const K::Vector_3 &e2, const K::Vector_3 &e3, const K::Point_3 &o)
{
	const auto p = point - o;
	return
	{
		e1.x() * p.x() + e1.y() * p.y() + e1.z() * p.z(),
		e2.x() * p.x() + e2.y() * p.y() + e2.z() * p.z(),
		e3.x() * p.x() + e3.y() * p.y() + e3.z() * p.z()
	};
};

static inline K::Point_2 get_point_2d(const K::Point_3 &p, const K::Vector_3 &e1, const K::Vector_3 &e2, const K::Vector_3 &e3, const K::Point_3 &o)
{
	const auto p_e = get_point_in_triangle_lcs(p, e1, e2, e3, o);
	assert(abs(p_e.z()) < EPS);
	return
	{
		p_e.x(),
		p_e.y()
	};
};

static inline K::Triangle_2 get_triangle_2d(const K::Triangle_3 &t, const K::Vector_3 &e1, const K::Vector_3 &e2, const K::Vector_3 &e3, const K::Point_3 &o)
{
	return
	{
		get_point_2d(t.vertex(0), e1, e2, e3, o),
		get_point_2d(t.vertex(1), e1, e2, e3, o),
		get_point_2d(t.vertex(2), e1, e2, e3, o)
	};
}

std::vector<K::Triangle_3> triangulate(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
	int i, const std::vector<K::Segment_3> &intersections, std::chrono::nanoseconds &triangulation_time)
{
#define CGAL_TRIANGLE_LCS 1

	std::vector<K::Triangle_3> out_triangles;

	auto begin = std::chrono::steady_clock::now();

	const auto &v0i = F(i, 0);
	const auto &v1i = F(i, 1);
	const auto &v2i = F(i, 2);

	K::Triangle_3 t
	{
		{ V(v0i, 0), V(v0i, 1), V(v0i, 2) },
		{ V(v1i, 0), V(v1i, 1), V(v1i, 2) },
		{ V(v2i, 0), V(v2i, 1), V(v2i, 2) }
	};

#if CGAL_TRIANGLE_LCS
	const auto &plane = t.supporting_plane();
#else
	K::Vector_3 e1, e2, e3;
	K::Point_3 o;
	get_triangle_lcs(t, e1, e2, e3, o);

	std::map<const K::Point_2, const K::Point_3 *> map2dto3d;
#endif

	std::vector<K::Segment_2> segments_2d;
	segments_2d.reserve(intersections.size());
	for (const auto &segment_3d : intersections)
	{
#if CGAL_TRIANGLE_LCS
		const auto &s = segments_2d.emplace_back(plane.to_2d(segment_3d.start()), plane.to_2d(segment_3d.end()));
#else
		const auto &s = segments_2d.emplace_back(get_point_2d(segment_3d.start(), e1, e2, e3, o), get_point_2d(segment_3d.end(), e1, e2, e3, o));
		map2dto3d[s.start()] = &segment_3d.start();
		map2dto3d[s.end()] = &segment_3d.end();
#endif
	}

	std::vector<K::Point_2> vertices_2d;
	vertices_2d.reserve(3);
	for (int vi = 0; vi < 3; ++vi)
	{
#if CGAL_TRIANGLE_LCS
		const auto &v = vertices_2d.emplace_back(plane.to_2d(t.vertex(vi)));
#else
		const auto &v = vertices_2d.emplace_back(get_point_2d(t.vertex(vi), e1, e2, e3, o));
		map2dto3d[v] = &t.vertex(vi);
#endif
	}

	T triangulation;
	std::vector<K::Segment_3> handled_constraints;

#if TRY_CATCH
	try
	{
#endif
		triangulation.insert(vertices_2d.begin(), vertices_2d.end());
		for (const auto &s : segments_2d)
		{
			triangulation.insert_constraint(s.start(), s.end());
#if CGAL_TRIANGLE_LCS
			handled_constraints.emplace_back(plane.to_3d(s.start()), plane.to_3d(s.end()));
#else
			handled_constraints.emplace_back(*map2dto3d.at(s.start()), *map2dto3d.at(s.end()));
#endif
		}
		//triangulation.insert_constraints(segments_2d.begin(), segments_2d.end());

#if TRY_CATCH
	}
	catch (...)
	{
		std::cout << "TRIANGULATION ERROR:" << std::endl <<
			i << std::endl <<
			t.vertex(0) << std::endl <<
			t.vertex(1) << std::endl <<
			t.vertex(2) << std::endl <<
			"LAST CONSTRAINT: " << std::endl;

		auto shift = t.supporting_plane().orthogonal_direction().to_vector();

		int id = viewer.append_mesh();
		draw_edges(t, id, TRIANGULATION_ERROR_COLOR);
		const auto last = (int)handled_constraints.size() - 1;
		for (int ci = 0; ci <= last; ++ci)
		{
			const auto color = ci == last ? TRIANGULATION_ERROR_COLOR : TRIANGULATION_HANDLED_COLOR;
			const auto &constraint = handled_constraints[ci];
			std::cout << constraint.start() << " -> " << constraint.end() << std::endl;
			if (ci == last)
			{
				draw_point(constraint.start(), id, color);
				draw_point(constraint.end(), id, color);
			}
			draw_edge(constraint.start(), constraint.end(), id, color);
		}

	}
#endif

	auto end = std::chrono::steady_clock::now();
	triangulation_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

	int id = viewer.append_mesh();
	//assert(triangulation.is_valid());
	for (auto iter = triangulation.finite_faces_begin(); iter != triangulation.finite_faces_end(); ++iter)
	{
		T::Face &face = *iter;

		const K::Triangle_3 new_triangle
		{
#if CGAL_TRIANGLE_LCS
			plane.to_3d(face.vertex(0)->point()),
			plane.to_3d(face.vertex(1)->point()),
			plane.to_3d(face.vertex(2)->point())
#else
			*map2dto3d.at(face.vertex(0)->point()),
			*map2dto3d.at(face.vertex(1)->point()),
			*map2dto3d.at(face.vertex(2)->point())
#endif
		};

		//assert(is_valid(new_triangle);
		if (is_valid(new_triangle))
			out_triangles.emplace_back(new_triangle);
	}
	return out_triangles;
}

template<typename Function>
void for_each_triangle(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Function &func)
{
	for (Eigen::Index row = 0; row < F.rows(); ++row)
	{
		const auto &v0i = F(row, 0);
		const auto &v1i = F(row, 1);
		const auto &v2i = F(row, 2);

		const K::Triangle_3 t
		{
			{ V(v0i, 0), V(v0i, 1), V(v0i, 2) },
			{ V(v1i, 0), V(v1i, 1), V(v1i, 2) },
			{ V(v2i, 0), V(v2i, 1), V(v2i, 2) }
		};

		func(t, row);
	}
}

bool is_inside_mesh(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const K::Triangle_3 &t)
{
	constexpr auto SHIFT = 1e10;

	const auto &n = get_normal(t);
	if (isinf(abs(n.x())) || isinf(abs(n.y())) || isinf(abs(n.z())) ||
		isnan(n.x()) || isnan(n.y()) || isnan(n.z()))
	{
		//assert(!"infinity or nan triangle normal");
		return false;
	}
	const auto &o = CGAL::centroid(t);
	size_t intersection_count = 0;

#if 1
	const K::Ray_3 r{ o, n };
	for_each_triangle(V, F, [&r, &intersection_count](const K::Triangle_3 &mesh_t, auto)
		{
			if (const auto result = CGAL::intersection(r, mesh_t))
				++intersection_count;
		});
#endif

#if 0
	const K::Ray_3 r{ o - n * SHIFT, n };
	struct ProjectionOntoRay
	{
		double distance;
		boost::optional<double> second_distance;

		inline ProjectionOntoRay(double d) :
			distance(d)
		{}
		inline ProjectionOntoRay(double d1, double d2) :
			distance(d1),
			second_distance(d2)
		{}

		bool operator<(const ProjectionOntoRay &other) const
		{
			const auto &max_distance = second_distance.has_value() ? second_distance.value() : distance;
			const auto &max_other_distance = other.second_distance.has_value() ? other.second_distance.value() : other.distance;

			return max_distance < max_other_distance;
		}
	};

	std::set<ProjectionOntoRay> projections;
	for_each_triangle(V, F, [&r, &projections](const K::Triangle_3 &mesh_t, auto)
		{
			if (const auto result = CGAL::intersection(r, mesh_t))
			{
				if (const auto *point = boost::get<K::Point_3>(&*result))
				{
					const auto d = length(r.start() - *point);
					projections.emplace(d);
				}
				else if (const auto *segment = boost::get<K::Segment_3>(&*result))
				{
					const auto d1 = length(r.start() - segment->start());
					const auto d2 = length(r.start() - segment->end());

					projections.emplace(d1, d2);
				}
			}
		});

	const ProjectionOntoRay check{ SHIFT };
	for (const auto &projection : projections)
	{
		if (projection < check)
			++intersection_count;
		else
			break;
	}
#endif

	return intersection_count % 2;
}

static inline void add_to_mesh(const K::Triangle_3 &t, Eigen::MatrixXd &V, Eigen::MatrixXi &F)
{
	const auto v = (Eigen::MatrixXd(3, 3) <<
		t.vertex(0).x(), t.vertex(0).y(), t.vertex(0).z(),
		t.vertex(1).x(), t.vertex(1).y(), t.vertex(1).z(),
		t.vertex(2).x(), t.vertex(2).y(), t.vertex(2).z()).finished();

	const auto f = (Eigen::MatrixXi(1, 3) <<
		0, 1, 2).finished();

	merge(V, F, v, f);
}

void create_mesh(const Eigen::MatrixXd &VA, const Eigen::MatrixXi &FA, const Eigen::MatrixXd &VB, const Eigen::MatrixXi &FB, Eigen::MatrixXd &VC, Eigen::MatrixXi &FC,
	const std::map<std::pair<int, bool>, std::vector<K::Triangle_3>> &new_triangles)
{
	assert(!FC.rows() && !VC.rows());

	for_each_triangle(VB, FB, [&new_triangles, &VA, &FA, &VC, &FC](const K::Triangle_3 &tb, int i)
		{
			const auto iter = new_triangles.find(std::make_pair(i, false));
			if (iter != new_triangles.end())
			{
				for (const auto &t : iter->second)
				{
					if (is_inside_mesh(VA, FA, t))
						add_to_mesh(t, VC, FC);
				}
			}
			else
			{
				if (is_inside_mesh(VA, FA, tb))
					add_to_mesh(tb, VC, FC);
			}
		});

	for_each_triangle(VA, FA, [&new_triangles, &VB, &FB, &VC, &FC](const K::Triangle_3 &ta, int i)
		{
			const auto iter = new_triangles.find(std::make_pair(i, true));
			if (iter != new_triangles.end())
			{
				for (const auto &t : iter->second)
				{
					if (is_inside_mesh(VB, FB, t))
						add_to_mesh(t, VC, FC);
				}
			}
			else
			{
				if (is_inside_mesh(VB, FB, ta))
					add_to_mesh(ta, VC, FC);
			}
		});
}

int intersect(const Eigen::MatrixXd &VA, const Eigen::MatrixXi &FA, const Eigen::MatrixXd &VB, const Eigen::MatrixXi &FB, Eigen::MatrixXd &VC, Eigen::MatrixXi &FC)
{
	auto triangulation_time = std::chrono::nanoseconds::zero();

	const auto intersections = calculate_intersections(VA, FA, VB, FB);

	std::map<std::pair<int, bool>, std::vector<K::Triangle_3>> new_triangles;
	for (const auto &[triangle_id, intersection] : intersections)
		new_triangles.emplace(triangle_id, triangulate(triangle_id.second ? VA : VB, triangle_id.second ? FA : FB, triangle_id.first, intersection, triangulation_time));

	create_mesh(VA, FA, VB, FB, VC, FC, new_triangles);

	write_time("triangulation time: ", triangulation_time);
	return 0;
}

void update(igl::opengl::glfw::Viewer &viewer)
{
	std::cout << "..." << std::endl;
	auto begin = std::chrono::steady_clock::now();

#if IGL_ALG
	igl::copyleft::cgal::mesh_boolean(VA, FA, VB, FB, igl::MESH_BOOLEAN_TYPE_INTERSECT, VC, FC, J);
#else
	const int error = intersect(VA, FA, VB, FB, VC, FC);
#endif

	auto end = std::chrono::steady_clock::now();
	auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
	write_time("time: ", elapsed_ms);

	viewer.data().clear();
	if (-1 == a_id)
		a_id = viewer.append_mesh();
	else
		viewer.data(a_id).clear();
	viewer.data(a_id).set_mesh(VA, FA);
	if (-1 == b_id)
		b_id = viewer.append_mesh();
	else
		viewer.data(b_id).clear();
	viewer.data(b_id).set_mesh(VB, FB);
	if (FC.count())
	{
		Eigen::MatrixXd C(FC.rows(), 3);
		for (size_t f = 0; f < (size_t)C.rows(); f++)
			C.row(f) = Eigen::RowVector3d(1, 1, 1);
		if (-1 == c_id)
			c_id = viewer.append_mesh(false);
		else
			viewer.data(c_id).clear();
		viewer.data().set_mesh(VC, FC);
		viewer.data().set_colors(C);
	}
}

int main(int argc, char *argv[])
{
	using namespace Eigen;

	if (2 == argc)
	{
		igl::read_triangle_mesh(argv[1], VA, FA);

		igl::opengl::glfw::Viewer viewer;
		viewer.data().set_mesh(VA, FA);
		viewer.data().set_face_based(true);
		viewer.launch();
	}
	else if (3 == argc || SimpleObjects::None != SIMPLE_OBJECTS)
	{
		const std::string path1 = SimpleObjects::None == SIMPLE_OBJECTS ? argv[1] : "";
		const std::string path2 = SimpleObjects::None == SIMPLE_OBJECTS ? argv[2] : "";

		switch (SIMPLE_OBJECTS)
		{
		case SimpleObjects::None:
			std::cout << path1 << std::endl << path2 << std::endl;
			break;

		case SimpleObjects::Cube:
			VA = (Eigen::MatrixXd(8, 3) <<
				0.0, 0.0, 0.0,
				0.0, 0.0, 1.0,
				0.0, 1.0, 0.0,
				0.0, 1.0, 1.0,
				1.0, 0.0, 0.0,
				1.0, 0.0, 1.0,
				1.0, 1.0, 0.0,
				1.0, 1.0, 1.0).finished();
			FA = (Eigen::MatrixXi(12, 3) <<
				1, 7, 5,
				1, 3, 7,
				1, 4, 3,
				1, 2, 4,
				3, 8, 7,
				3, 4, 8,
				5, 7, 8,
				5, 8, 6,
				1, 5, 6,
				1, 6, 2,
				2, 6, 8,
				2, 8, 4).finished().array() - 1;

			VB = (Eigen::MatrixXd(8, 3) <<
				-0.1, -0.1, -0.1,
				-0.1, -0.1, 0.5,
				-0.1, 1.1, -0.1,
				-0.1, 1.1, 0.5,
				1.1, -0.1, -0.1,
				1.1, -0.1, 0.5,
				1.1, 1.1, -0.1,
				1.1, 1.1, 0.5).finished();
			FB = (Eigen::MatrixXi(12, 3) <<
				1, 7, 5,
				1, 3, 7,
				1, 4, 3,
				1, 2, 4,
				3, 8, 7,
				3, 4, 8,
				5, 7, 8,
				5, 8, 6,
				1, 5, 6,
				1, 6, 2,
				2, 6, 8,
				2, 8, 4).finished().array() - 1;
			break;

		case SimpleObjects::Triangles:
			VA = (Eigen::MatrixXd(3, 3) <<
				0.02465, 0.000499, 0.027,
				0.027148, -0.003825, 0.01494,
				0.03715, 0.003849, 0.01494).finished();

			VB = (Eigen::MatrixXd(3, 3) <<
				-0.002, -0.002, 0.1,
				-0.002, -0.002, 0,
				0.098, -0.002, 0).finished();

			FA = (Eigen::MatrixXi(1, 3) <<
				0, 1, 2).finished();
			FB = (Eigen::MatrixXi(1, 3) <<
				0, 1, 2).finished();

			break;
		}

		if (SimpleObjects::None != SIMPLE_OBJECTS || igl::read_triangle_mesh(path1, VA, FA) && igl::read_triangle_mesh(path2, VB, FB))
		{
			update(viewer);

			viewer.data().show_lines = true;
			viewer.callback_key_pressed = [](igl::opengl::glfw::Viewer &viewer, unsigned key, int modifiers)
			{
				enum class Visibility : uint8_t
				{
					All,
					A,
					B,
					C,
					None,

					Last = None,
					First = All
				};

				static Visibility visibility = Visibility::All;

				switch(key)
				{
				case 'H':
				case 'h':
					visibility = Visibility::Last == visibility ? Visibility::First : (Visibility)((uint8_t)visibility + 1);
					if (-1 != a_id)
						viewer.data(a_id).set_visible(Visibility::All == visibility || Visibility::A == visibility);
					if (-1 != b_id)
						viewer.data(b_id).set_visible(Visibility::All == visibility || Visibility::B == visibility);
					if (-1 != c_id)
						viewer.data(c_id).set_visible(Visibility::C == visibility);

					return true;

				case 'Q':
				case 'q':
					viewer.core().camera_dnear -= 0.1;
					return true;

				case 'E':
				case 'e':
					viewer.core().camera_dnear += 0.1;
					return true;

				default:
					return false;
				}
			};
			viewer.launch();
		}
	}
	return 0;
}
