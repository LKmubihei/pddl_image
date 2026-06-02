(define (problem picture_190)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery green_battery red_battery red_pump green_regulator green_regulator_1 - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear green_battery)
        (clear red_battery)
        (clear green_regulator_1)
        (clear green_regulator)
        (clear blue_battery)
        (on green_regulator_1 red_pump)
        (part_at red_pump table)
        (part_at green_battery table)
        (part_at red_battery table)
        (part_at blue_battery battery_placement)
        (part_at green_regulator regulator_placement)
    )
    
    (:goal
        (and
            (part_at red_pump pump_placement)
        )
    )
)