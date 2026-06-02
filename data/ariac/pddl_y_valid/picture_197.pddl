(define (problem picture_193)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        green_pump green_regulator green_battery red_regulator blue_battery green_regulator_1 - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear green_pump)
        (clear green_regulator)
        (clear green_battery)
        (clear green_regulator_1)
        (clear blue_battery)
        (clear red_regulator)
        (part_at green_pump table)
        (part_at green_regulator regulator_placement)
        (part_at green_battery table)
        (part_at green_regulator_1 table)
        (part_at blue_battery battery_placement)
        (part_at red_regulator table)
    )
    
    (:goal
        (and
            (part_at green_pump pump_placement)
        )
    )
)