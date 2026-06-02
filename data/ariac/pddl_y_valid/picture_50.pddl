(define (problem picture_50)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        green_battery red_sensor green_regulator - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear green_battery)
        (clear red_sensor)
        (clear green_regulator)
        (on green_battery red_sensor)
        (part_at red_sensor table)
        (part_at green_regulator table)
    )
    
    (:goal
        (and
            (part_at green_regulator regulator_placement)
            (part_at green_battery battery_placement)
        )
    )
)
