(define (problem picture_193)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        green_pump red_pump green_regulator green_battery  blue_battery  - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear green_pump)
        (clear green_regulator)
        (clear green_battery)
        (clear blue_battery)
        (clear red_pump)
        (part_at green_pump table)
        (part_at red_pump pump_placement)
        (part_at green_regulator table)
        (part_at green_battery table)
        (part_at blue_battery table)
    )
    
    (:goal
        (and

            (part_at green_regulator regulator_placement)
            (part_at blue_battery battery_placement)
        )
    )
)