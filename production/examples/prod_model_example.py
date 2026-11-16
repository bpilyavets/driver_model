import labor_cost as lca

source = lca.SourceSchema(
    dimensions={
        "grade": "grade",
        "reg_coef_distr": "reg_coef",
    },
)

populations = (
    lca.Population(
        "permanent",
        {"motivation_type": "permanent"},
    ),
    lca.Population(
        "gpd",
        {"motivation_type": "gpd"},
    ),
    lca.Population(
        "pkc",
        {"motivation_type": "pkc"},
    ),
)

fields = (
    lca.FieldSpec(
        "permanent_rate", "base_stavka",
        population="permanent",
        units="currency/time", constraint="nonnegative",
    ),
    lca.FieldSpec(
        "vyrabotka", "vyrabotka_act",
        population="permanent",
        units="time", constraint="nonnegative",
    ),
    lca.FieldSpec(
        "reg_coefficient", "reg_coef",
        population="permanent",
        units="dimensionless", constraint="nonnegative"
    ),
    lca.FieldSpec(
        "sick_leave", "sum_pay_bl_reg_base",
        population="permanent",
        units="currency", constraint="signed",
    ),
    lca.FieldSpec(
        "vacation_pay", "sum_pay_otpusk_reg_base",
        population="permanent",
        units="currency", constraint="signed",
    ),
    lca.FieldSpec(
        "holiday_pay", "sum_pay_prazdnik_reg_base",
        population="permanent",
        units="currency", constraint="signed",
    ),
    lca.FieldSpec(
        "night_pay", "sum_pay_oklad_night_reg_base",
        population="permanent",
        units="currency", constraint="signed"
    ),
    lca.FieldSpec(
        "meal_compensation", "sum_pay_compens_meal_reg_base",
        population="permanent",
        units="currency", constraint="signed"
    ),
    lca.FieldSpec(
        "other_pay", "sum_pay_compacted",
        population="permanent",
        units="currency", constraint="signed"
    ),
    lca.FieldSpec(
        "yearly_bonus", "sum_pay_gb_reg_base",
        population="permanent",
        units="currency", constraint="signed"
    ),
    lca.FieldSpec(
        "bonus_kpi", "bonus_kpi_comp",
        population="permanent",
        units="currency", constraint="signed",
    ),
    lca.FieldSpec(
        "bonus_sharing", "bonus_sharing_comp",
        population="permanent",
        units="currency", constraint="signed",
    ),
    lca.FieldSpec(
        "bonus_quarterly", "bonus_quarterly_comp_reg_base",
        population="permanent",
        units="currency", constraint="signed",
    ),
    lca.FieldSpec(
        "bonus_overtime", "bonus_overtime_comp_reg_base",
        population="permanent",
        units="currency", constraint="signed",
    ),
    lca.FieldSpec(
        "gpd_minutes", "all_effect_min_gpd",
        population="gpd",
        units="time", constraint="nonnegative",
    ),
    lca.FieldSpec(
        "gpd_rate", "gpd_adjusted_minute_rate",
        population="gpd",
        units="currency/time", constraint="positive",
    ),
    lca.FieldSpec(
        "gpd_adj_coeff", "gpd_adj_coeff",
        population="gpd",
        units="currency/time", constraint="nonnegative",
    ),
    lca.FieldSpec(
        "pkc_pay", "pkc_pay",
        population="pkc",
        units="currency", constraint="nonnegative",
    ),
)


factors = lca.outer_factors() + (
    # Permanent/temporary shares within each team.
    lca.Factor(
        "motivation_mix",
        "population_mix",
        dimensions=("motivation_kind",),
        members=("permanent", "gpd", "pkc"),
    ),

    # Grade shares within each team's permanent population.
    lca.Factor(
        "region_mix",
        "distribution",
        population="permanent",
        dimensions=("reg_coef_distr",),
    ),

    lca.Factor(
        "grade_within_region_mix",
        "distribution",
        population="permanent",
        dimensions=("grade",),
        given=("reg_coef_distr",),
        fallbacks=(()),
    ),

    # Which permanent employees contribute within each cell.
    lca.Factor(
        "permanent_members",
        "membership",
        population="permanent",
        dimensions=("grade", "reg_coef_distr"),
        fallbacks=(()),
    ),

    # Preserve permanent rate/hour pairing by employee.
    lca.Factor(
        "permanent_rate",
        "aligned",
        population="permanent",
        source="permanent_rate",
    ),
    # чтобы можно было умножать на ргекэффы, а не токльо использовать их как разрез
    lca.Factor(
        "regional_policy",
        "policy",
        population="permanent",
        dimensions=("reg_coef_distr"),
        source="reg_coefficient"
    ),
    lca.Factor(
        "vyrabotka",
        "aligned",
        population="permanent",
        source="permanent_hours",
    ),

    lca.Factor(
        "sick_leave",
        "payment",
        population="permanent",
        dimensions=("reg_coef_distr", "grade"),
        formula=lca.field("sick_leave"),
        fallbacks=(()),
    ),

    lca.Factor(
        "vacation_pay",
        "payment",
        population="permanent",
        dimensions=("reg_coef_distr", "grade"),
        formula=lca.field("vacation_pay"),
        fallbacks=(()),
    ),

    lca.Factor(
        "holiday_pay",
        "payment",
        population="permanent",
        dimensions=("reg_coef_distr", "grade"),
        formula=lca.field("holiday_pay"),
        fallbacks=(()),
    ),

    lca.Factor(
        "night_pay",
        "payment",
        population="permanent",
        dimensions=("reg_coef_distr", "grade"),
        formula=lca.field("night_pay"),
        fallbacks=(()),
    ),

    lca.Factor(
        "meal_compensation",
        "payment",
        population="permanent",
        dimensions=("reg_coef_distr", "grade"),
        formula=lca.field("meal_compensation"),
        fallbacks=(()),
    ),

    lca.Factor(
        "yearly_bonus",
        "payment",
        population="permanent",
        dimensions=("reg_coef_distr", "grade"),
        formula=lca.field("yearly_bonus"),
        fallbacks=(()),
    ),

    lca.Factor(
        "bonus_quarterly",
        "payment",
        population="permanent",
        dimensions=("reg_coef_distr", "grade"),
        formula=lca.field("bonus_quarterly"),
        fallbacks=(()),
    ),

    lca.Factor(
        "bonus_overtime",
        "payment",
        population="permanent",
        dimensions=("reg_coef_distr", "grade"),
        formula=lca.field("bonus_overtime"),
        fallbacks=(()),
    ),
    lca.Factor(
        "bonus_sharing",
        "payment",
        population="permanent",
        dimensions=("reg_coef_distr", "grade"),
        formula=lca.field("bonus_sharing"),
        fallbacks=(()),
    ),

    lca.Factor(
        "bonus_kpi",
        "payment",
        population="permanent",
        dimensions=("reg_coef_distr", "grade"),
        formula=lca.field("bonus_kpi"),
        fallbacks=(()),
    ),

    lca.Factor(
        "other_pay",
        "payment",
        population="permanent",
        dimensions=("reg_coef_distr", "grade"),
        formula=lca.field("other_pay"),
        fallbacks=(()),
    ),

    # Temporary workers have no categorical breakdown.
    lca.Factor(
        "gpd_members",
        "membership",
        population="gpd",
    ),

    # Preserve all four temporary inputs by employee.
    lca.Factor(
        "gpd_minutes",
        "aligned",
        population="gpd",
        source="gpd_minutes",
    ),
    lca.Factor(
        "gpd_rate",
        "aligned",
        population="gpd",
        source="gpd_rate",
    ),
    lca.Factor(
        "gpd_adj_coeff",
        "aligned",
        population="gpd",
        source="gpd_adj_coeff",
    ),

    lca.Factor(
        "pkc_members",
        "membership",
        population="pkc",
    ),
    lca.Factor(
        "pkc_pay",
        "payment",
        population="pkc",
        formula=lca.field("pkc_pay"),
        fallbacks=(()),
    ),
)


row_costs = (
    lca.RowCost(
        "permanent",
        lca.field("reg_coefficient") * (
            lca.field("permanent_rate") * lca.field("vyrabotka")
            + lca.field("sick_leave") + lca.field("vacation_pay")
            + lca.field("holiday_pay") + lca.field("night_pay")
            + lca.field("meal_compensation") + lca.field("yearly_bonus")
            + lca.field("bonus_quarterly") + lca.field("bonus_kpi")
        )
        + lca.field("bonus_sharing") + lca.field("bonus_overtime")
        + lca.field("other_pay"),
    ),
    lca.RowCost(
        "gpd",
        lca.field("gpd_rate") * lca.field("gpd_minutes") * lca.field("gpd_adj_coeff"),
    ),
    lca.RowCost(
        "pkc",
        lca.field("pkc_pay")
    )
)

permanent_cell_cost = (
    lca.value("regional_policy") * (
        lca.aligned_mean(
            lca.value("permanent_rate") * lca.value("vyrabotka"),
            "permanent_members",
        )
        + lca.value("sick_leave") + lca.value("vacation_pay")
        + lca.value("holiday_pay") + lca.value("night_pay")
        + lca.value("meal_compensation") + lca.value("yearly_bonus")
        + lca.value("bonus_quarterly") + lca.value("bonus_kpi")
    )
    + lca.value("bonus_sharing")+ lca.value("bonus_overtime")
    + lca.value("other_pay")
)

permanent_unit_cost = lca.sum_over(
    lca.value("region_mix") * lca.value("grade_within_region_mix") * permanent_cell_cost,
    "reg_coef_distr", "grade",
)

gpd_unit_cost = lca.aligned_mean(
    lca.value("gpd_minutes") * lca.value("gpd_adj_coeff") * lca.value("gpd_rate"),
    "gpd_members"
)

pkc_unit_cost = lca.aligned_mean(
    lca.value("pkc_pay"),
    "pkc_members"
)

permanent_game = lca.Game(
    name="permanent",
    population="permanent",
    formula=permanent_unit_cost,
    switches=(
        lca.Switch(
            "region_mix",
            label="Региональный микс",
        ),
        lca.Switch(
            "grade_mix_within_region",
            label="Микс грейдов внутри региона",
        ),
        lca.Switch(
            "permanent_members",
            label="Микс сотрудников внутри региона и грейда",
        ),
        lca.Switch(
            "vyrabotka",
            label="Выработка",
        ),
        lca.Switch(
            "sick_leave",
            label="Больничные",
        ),
        lca.Switch(
            "holiday_pay",
            label="Доплата за работу в праздники",
        ),
        lca.Switch(
            "vacation_pay",
            label="Отпускные",
        ),
        lca.Switch(
            "night_pay",
            label="Ночные",
        ),
        lca.Switch(
            "meal_compensation",
            label="Компенсация питания",
        ),
        lca.Switch(
            "yearly_bonus",
            label="Годовой бонус",
        ),
        lca.Switch(
            "bonus_kpi",
            label="Премия за kpi",
        ),
        lca.Switch(
            "bonus_overtime",
            label="Премия за овертаймы",
        ),
        lca.Switch(
            "bonus_sharing",
            label="Премия за шеринг",
        ),
        lca.Switch(
            "bonus_quarterly",
            label="Квартальная премия",
        ),
        lca.Switch(
            "other_pay",
            label="Прочие выплаты",
        ),
    ),
)

gpd_game = lca.Game(
    name="gpd",
    population="gpd",
    formula=gpd_unit_cost,
    switches=(
        lca.Switch(
            "gpd_members",
            label="Микс сотрудников ГПД",
        ),
        lca.Switch(
            "gpd_minutes",
            label="Отработанные минуты ГПД",
        ),
        lca.Switch(
            "gpd_rate",
            label="Средняя ставка ГПД",
        ),
        lca.Switch(
            "gpd_adj_coeff",
            label="Средний бонусный коэффициент ГПД",
        ),
    ),
)

pkc_game = lca.Game(
    name="pkc",
    population="pkc",
    formula=pkc_unit_cost,
    switches=(
        lca.Switch(
            "pkc_members",
            label="Микс сотрудников ПКЦ",
        ),
        lca.Switch(
            "pkc_pay",
            label="Выплаты ПКЦ",
        ),
    )
)

mix = lca.value("team_mix")

team_game = lca.Game(
    name="team",
    formula=(
        lca.select(mix, "motivation_kind", "permanent")
        * lca.child("permanent")
        + lca.select(mix, "motivation_kind", "gpd")
        * lca.child("gpd")
        + lca.select(mix, "motivation_kind", "pkc")
        * lca.child("pkc")
    ),
    switches=(
        lca.Switch(
            "motivation_mix",
            label="Микс по видам мотивации (ТК, ГПД, ПКЦ)",
        ),
        lca.Switch(
            "permanent_cost",
            child="permanent",
            label="Кост сотрудников на ТК",
        ),
        lca.Switch(
            "gpd_cost",
            child="gpd",
            label="Кост исполнителей ГПД",
        ),
        lca.Switch(
            "pkc_cost",
            child="pkc",
            label="Кост исполнителей ПКЦ",
        ),
    ),
)

model = lca.ModelSpec(
    model_id="simple-payroll",
    version="1.0.0",
    graph=lca.ModelGraph(
        source=source,
        populations=populations,
        fields=fields,
        row_costs=row_costs,
        factors=factors,
        games=(
            permanent_game,
            gpd_game,
            pkc_game,
            team_game,
        ),
        team_game="team",
        organization=lca.organization_game("team"),
        local=lca.local_game("team"),
        missing_economics="carry_observed",
    ),
)

compiled = lca.compile_model(model)