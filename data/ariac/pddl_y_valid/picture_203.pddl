(define (problem picture_203)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_battery red_battery_1 red_pump blue_regulator - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_battery)
        (clear red_battery_1)
        (on blue_regulator red_pump)
        (clear blue_regulator)
        (part_at red_pump table)
        (part_at red_battery battery_placement)
        (part_at red_battery_1 table)

    )
    
    (:goal
        (and
            (part_at red_pump pump_placement)
            (part_at blue_regulator regulator_placement)
        )
    )
)